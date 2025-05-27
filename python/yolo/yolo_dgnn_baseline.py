import os
import shutil
import json
import random
from tqdm import tqdm
import logging
import time
import psutil
import gc
from functools import partial
import numpy as np
import networkx as nx
from tabulate import tabulate
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import roi_align
from ultralytics import YOLO
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
from codecarbon import EmissionsTracker
from torch_geometric.nn import GCNConv

logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("ultralytics.yolo").setLevel(logging.ERROR)

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


final_output_json='/var/scratch/sismail/data/processed/final_annotations_without_occluded.json'
image_directory = '/var/scratch/sismail/data/images'

test_ratio = 0.2
valid_ratio = 0.1
random_seed = 42

with open(final_output_json, 'r') as f:
    annotations = json.load(f)

image_filenames = list(annotations['images'].keys())

random.seed(random_seed)
random.shuffle(image_filenames)

num_test = int(len(image_filenames) * test_ratio)
test_images = image_filenames[:num_test]
train_images = image_filenames[num_test:]
num_valid = int(len(train_images) * valid_ratio)
valid_images = train_images[:num_valid]
train_images = train_images[num_valid:]

train_annotations = {
    'all_parts': annotations['all_parts'],
    'images': {img_name: annotations['images'][img_name] for img_name in train_images}
}

valid_annotations = {
    'all_parts': annotations['all_parts'],
    'images': {img_name: annotations['images'][img_name] for img_name in valid_images}
}

test_annotations = {
    'all_parts': annotations['all_parts'],
    'images': {img_name: annotations['images'][img_name] for img_name in test_images}
}


class BikePartsDetectionDataset(Dataset):
    def __init__(self, annotations_dict, image_dir, transform=None, augment=True, target_size=(640, 640)):
        self.all_parts = annotations_dict['all_parts']
        self.part_to_idx = {part: idx for idx, part in enumerate(self.all_parts)}
        self.idx_to_part = {idx: part for idx, part in enumerate(self.all_parts)}
        self.image_data = annotations_dict['images']
        self.image_filenames = list(self.image_data.keys())
        self.image_dir = image_dir
        self.transform = transform
        self.augment = augment
        self.target_size = target_size

    def __len__(self):
        return len(self.image_filenames) * (2 if self.augment else 1)

    def apply_augmentation(self, image, boxes):
        if random.random() < 0.5:
            image = transforms.functional.hflip(image)
            w = image.width
            boxes = boxes.clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        if random.random() < 0.8:
            image = transforms.functional.adjust_brightness(image, brightness_factor=random.uniform(0.6, 1.4))
        if random.random() < 0.8:
            image = transforms.functional.adjust_contrast(image, contrast_factor=random.uniform(0.6, 1.4))
        if random.random() < 0.5:
            image = transforms.functional.adjust_saturation(image, saturation_factor=random.uniform(0.7, 1.3))

        return image, boxes

    def __getitem__(self, idx):
        real_idx = idx % len(self.image_filenames)
        do_augment = self.augment and (idx >= len(self.image_filenames))

        img_filename = self.image_filenames[real_idx]
        img_path = os.path.join(self.image_dir, img_filename)

        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size

        annotation = self.image_data[img_filename]
        available_parts_info = annotation['available_parts']
        missing_parts_names = annotation.get('missing_parts', [])

        boxes = []
        labels = []

        for part_info in available_parts_info:
            part_name = part_info['part_name']
            bbox = part_info['absolute_bounding_box']
            xmin = bbox['left']
            ymin = bbox['top']
            xmax = xmin + bbox['width']
            ymax = ymin + bbox['height']
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.part_to_idx[part_name])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if do_augment:
            image, boxes = self.apply_augmentation(image, boxes)

        image = transforms.functional.resize(image, self.target_size)
        new_width, new_height = self.target_size
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        image = transforms.functional.to_tensor(image)

        missing_labels = torch.tensor(
            [self.part_to_idx[part] for part in missing_parts_names],
            dtype=torch.int64
        )

        target = {
            'boxes': boxes,
            'labels': labels,
            'missing_labels': missing_labels,
            'image_id': torch.tensor([real_idx])
        }

        return image, target


train_dataset = BikePartsDetectionDataset(
    annotations_dict=train_annotations,
    image_dir=image_directory,
    augment=False
)

valid_dataset = BikePartsDetectionDataset(
    annotations_dict=valid_annotations,
    image_dir=image_directory,
    augment=False
)

test_dataset = BikePartsDetectionDataset(
    annotations_dict=test_annotations,
    image_dir=image_directory,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    worker_init_fn=seed_worker,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch))
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch))
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch))
)


class SpatialDGNN(torch.nn.Module):
    def __init__(self, feat_dim=256, hidden_dim=512):
        super().__init__()
        in_dim = feat_dim + 6
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, in_dim)

    def forward(self, x, edge_index, edge_weight=None):
        h = nn.functional.relu(self.conv1(x, edge_index, edge_weight))
        return self.conv2(h, edge_index, edge_weight)

def save_detection_graph(
    np_image,
    boxes,
    W,
    edge_index,
    output_path: str,
    title: str = None,
    dpi: int = 150
):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    centers = ((boxes[:, :2] + boxes[:, 2:]) / 2).cpu().numpy()

    G = nx.Graph()
    for i, (x, y) in enumerate(centers):
        G.add_node(i, pos=(x, y))

    for src, dst in edge_index.t().cpu().numpy():
        w = float(W[src, dst].cpu())
        G.add_edge(int(src), int(dst), weight=w)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    ax.imshow(np_image)
    ax.axis('off')

    pos = nx.get_node_attributes(G, 'pos')
    weights = [d['weight'] for _, _, d in G.edges(data=True)]

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=[2.0 * w for w in weights],
        alpha=0.7,
        edge_color='yellow'
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=50,
        node_color='red'
    )
    if title:
        ax.set_title(title)

    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)



def part_level_evaluation(results, part_to_idx, idx_to_part):
    parts = list(part_to_idx.values())

    Y_true = np.array([[1 if p in r['true_missing_parts'] else 0 for p in parts] for r in results])
    Y_pred = np.array([[1 if p in r['predicted_missing_parts'] else 0 for p in parts] for r in results])

    micro_f1 = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(Y_true, Y_pred, average='macro', zero_division=0)

    FN = np.logical_and(Y_true==1, Y_pred==0).sum()
    TP = np.logical_and(Y_true==1, Y_pred==1).sum()
    FP = np.logical_and(Y_true==0, Y_pred==1).sum()

    N_images = len(results)
    miss_rate = FN/(FN+TP) if (FN+TP)>0 else 0
    fppi = FP/N_images

    overall_acc = accuracy_score(Y_true.flatten(), Y_pred.flatten())
    overall_prec = precision_score(Y_true.flatten(), Y_pred.flatten(), zero_division=0)
    overall_rec = recall_score(Y_true.flatten(), Y_pred.flatten(), zero_division=0)
    overall_f1 = f1_score(Y_true.flatten(), Y_pred.flatten(), zero_division=0)
    print(f"[METRIC] Micro-F1: {micro_f1:.4f}")
    print(f"[METRIC] Macro-F1: {macro_f1:.4f}")
    print(f"[METRIC] Miss Rate: {miss_rate:.4f}")
    print(f"[METRIC] FPPI: {fppi:.4f}")
    print(f"[METRIC] Overall Acc: {overall_acc:.4f}")
    print(f"[METRIC] Precision: {overall_prec:.4f}")
    print(f"[METRIC] Recall: {overall_rec:.4f}")
    print(f"[METRIC] F1: {overall_f1:.4f}")

    table=[]
    for j,p in enumerate(parts):
        acc = accuracy_score(Y_true[:,j], Y_pred[:,j])
        prec = precision_score(Y_true[:,j], Y_pred[:,j], zero_division=0)
        rec = recall_score(Y_true[:,j], Y_pred[:,j], zero_division=0)
        f1s = f1_score(Y_true[:,j], Y_pred[:,j], zero_division=0)
        table.append([idx_to_part[p], f"{acc:.3f}", f"{prec:.3f}", f"{rec:.3f}", f"{f1s:.3f}"])

    print("[METRIC-TABLE] Per-Part Evaluation")
    print(tabulate(table, headers=["Part","Acc","Prec","Rec","F1"], tablefmt="fancy_grid"))

def run_inference(yolo_model, dgnn, reducer, loader, part_to_idx, idx_to_part, device,
    sigma_spatial=20.0, gamma_appear=5.0, weight_threshold=1e-3, conf_threshold=0.5,
    stride=32, save_vis=False, vis_dir='/home/sismail/Thesis/visualisations', max_vis=3):

    yolo_model.eval()
    dgnn.eval()
    reducer.eval()
    results = []
    alpha = 1.0 / (2 * sigma_spatial**2)
    num_vis = 0

    stored_feats = []
    detect_layer = yolo_model.model.model[-2]
    hook = detect_layer.register_forward_hook(lambda m, inp, out: stored_feats.append(out))

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="DGNN Inference"):
            stored_feats.clear()
            np_imgs = [(img.cpu().permute(1,2,0).numpy()*255).astype('uint8') for img in images]
            preds = yolo_model(np_imgs, device=device, verbose=False)

            for i, (feat_tensor, det, target) in enumerate(zip(stored_feats, preds, targets)):
                feats = feat_tensor.to(device)
                boxes = det.boxes.xyxy.to(device)
                confs = det.boxes.conf.to(device).unsqueeze(1)
                clss  = det.boxes.cls.to(device).unsqueeze(1)
                num_boxes = boxes.size(0)

                cxs = ((boxes[:,0]+boxes[:,2])/2).unsqueeze(1)
                cys = ((boxes[:,1]+boxes[:,3])/2).unsqueeze(1)
                ws  = (boxes[:,2]-boxes[:,0]).unsqueeze(1)
                hs  = (boxes[:,3]-boxes[:,1]).unsqueeze(1)

                if num_boxes > 0:
                    boxes_scaled = boxes / stride
                    batch_idx = torch.zeros(num_boxes,1,device=device)
                    roi_boxes = torch.cat([batch_idx, boxes_scaled], dim=1)
                    pooled = roi_align(feats, roi_boxes, output_size=(1,1))
                    pooled = pooled.view(num_boxes, -1)
                    pooled = reducer(pooled)

                    x = torch.cat([cxs, cys, ws, hs, confs, clss, pooled], dim=1)

                    centers = torch.cat([cxs, cys], dim=1)
                    D = torch.cdist(centers, centers)
                    Wsp = torch.exp(-alpha * D**2)
                    S = nn.functional.cosine_similarity(pooled.unsqueeze(1),
                                                         pooled.unsqueeze(0), dim=2)
                    Wap = torch.exp(gamma_appear * S)
                    W   = Wsp * Wap

                    src, dst = ( W > weight_threshold).nonzero(as_tuple=True)
                    if src.numel() > 0:
                        edge_index = torch.stack([src,dst],dim=0)
                        edge_weight= W[src,dst]
                        x_ref = dgnn(x, edge_index, edge_weight)
                        refined_confs = x_ref[:,4].cpu()
                        refined_cls   = x_ref[:,5].round().clamp(0,len(part_to_idx)-1).long().cpu()
                    else:
                        refined_confs = confs.squeeze(1).cpu()
                        refined_cls   = clss.squeeze(1).cpu().long()
                else:
                    refined_confs = torch.tensor([],device='cpu')
                    refined_cls   = torch.tensor([],dtype=torch.long,device='cpu')

                if save_vis and num_vis < max_vis:
                    save_detection_graph(
                        images[i].cpu().permute(1,2,0).numpy(),
                        boxes.cpu(), W if num_boxes>0 else None,
                        edge_index if num_boxes>0 and src.numel()>0 else None,
                        os.path.join(vis_dir,f"img{target['image_id'].item():04d}.png")
                    )
                    num_vis+=1

                keep = (refined_confs>=conf_threshold).nonzero().squeeze(1)
                pred_labels = set(refined_cls[keep].tolist())
                true_missing= set(target['missing_labels'].tolist())
                all_parts = set(part_to_idx.values())

                results.append({
                    'image_id': target['image_id'].item(),
                    'predicted_missing_parts': all_parts - pred_labels,
                    'true_missing_parts': true_missing
                })

    hook.remove()
    return results

def train_and_validate(yolo_backbone='yolov11n.pt', project_dir='/var/scratch/sismail/models/yolo/runs', run_name='bikeparts_dgnn_euclid',
    epochs=50, patience=5, lr_yolo=1e-4, lr_gnn=1e-4, weight_decay=1e-4, device='cuda'):
    yolo = YOLO(yolo_backbone, verbose=False).to(device)
    optim_yolo = torch.optim.AdamW(yolo.model.parameters(), lr=lr_yolo, weight_decay=weight_decay)

    dgnn = SpatialDGNN().to(device)
    reducer = nn.Linear(576,256).to(device)
    optim_gnn = torch.optim.AdamW(
        list(dgnn.parameters())+list(reducer.parameters()),
        lr=lr_gnn, weight_decay=weight_decay)

    best_f1, wait = 0.0, 0
    if torch.cuda.is_available():
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)

    for epoch in range(1, epochs+1):
        yolo.model.train()
        dgnn.train()
        reducer.train()
        running_loss = 0.0

        batch_times = []
        gpu_memories = []
        cpu_memories = []
        stored_feats = []

        hook = yolo.model.model[-2].register_forward_hook(lambda m,i,o: stored_feats.append(o))

        with EmissionsTracker(log_level="critical", save_to_file=False) as tracker:
            for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                np_imgs = [(img.cpu().permute(1,2,0).numpy()*255).astype('uint8') for img in images]
                start_time = time.time()

                preds = yolo(np_imgs, device=device, verbose=False)
                imgs_tensor = torch.stack(images, dim=0).to(device)
                batch = {"img": imgs_tensor, **targets}
                det_loss = yolo.model.loss(batch, preds)

                total_gnn = 0.0
                for feats, det, tgt in zip(stored_feats, preds, targets):
                    feats = feats.to(device)
                    boxes = det.boxes.xyxy.to(device)
                    confs = det.boxes.conf.to(device).unsqueeze(1)
                    clss  = det.boxes.cls.to(device).unsqueeze(1)
                    cxs = ((boxes[:,0]+boxes[:,2])/2).unsqueeze(1)
                    cys = ((boxes[:,1]+boxes[:,3])/2).unsqueeze(1)
                    ws = (boxes[:,2]-boxes[:,0]).unsqueeze(1)
                    hs = (boxes[:,3]-boxes[:,1]).unsqueeze(1)
                    num = boxes.size(0)
                    if num<2: continue

                    batch_idx = torch.zeros(num,1,device=device)
                    roi_boxes = torch.cat([batch_idx, boxes/32],dim=1)
                    pool = roi_align(feats, roi_boxes, output_size=(1,1)).view(num,-1)
                    reduced = reducer(pool)

                    x = torch.cat([cxs,cys,ws,hs,confs,clss,reduced],dim=1)
                    centers = torch.cat([cxs,cys],dim=1)
                    D = torch.cdist(centers,centers)
                    Wsp = torch.exp(- (1/(2*20.0**2)) * D**2)
                    S = nn.functional.cosine_similarity(reduced.unsqueeze(1), reduced.unsqueeze(0), dim=2)
                    Wap = torch.exp(5.0 * S)
                    W = Wsp * Wap
                    src,dst = (W>1e-3).nonzero(as_tuple=True)
                    if src.numel()==0: continue

                    eidx = torch.stack([src,dst],dim=0)
                    ewei = W[src,dst]
                    x_ref = dgnn(x,eidx,ewei)
                    total_gnn += nn.functional.mse_loss(x_ref[:,-256:], x[:,-256:])

                if len(preds)>0: total_gnn /= len(preds)

                optim_yolo.zero_grad()
                optim_gnn.zero_grad()
                loss = det_loss + 0.1 * total_gnn
                loss.backward()
                optim_gnn.step()
                optim_yolo.step()

                running_loss += loss.item()
                

                end_time = time.time()
                inference_time = end_time - start_time
                batch_times.append(inference_time)

                if torch.cuda.is_available():
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    gpu_mem_used = mem_info.used / (1024 ** 2)
                    gpu_memories.append(gpu_mem_used)
                else:
                    gpu_mem_used = 0

                stored_feats.clear()
                gc.collect()

                cpu_mem_used = psutil.virtual_memory().used / (1024 ** 2)
                cpu_memories.append(cpu_mem_used)

            print(f"Epoch {epoch}, Train Loss={running_loss/len(train_loader):.4f}, time (s): {inference_time:.3f}, \
                  GPU Mem (MB): {gpu_mem_used:.0f}, CPU Mem (MB): {cpu_mem_used:.0f}")
            hook.remove()


            val_res = run_inference(yolo, dgnn, reducer, valid_loader,
                                    train_dataset.part_to_idx, train_dataset.idx_to_part,
                                    device, save_vis=False)
            parts = list(train_dataset.part_to_idx.values())
            Y_True = np.array([[1 if p in r['true_missing_parts'] else 0 for p in parts] for r in val_res])
            Y_Pred = np.array([[1 if p in r['predicted_missing_parts'] else 0 for p in parts] for r in val_res])
            macro_F1 = f1_score(Y_True, Y_Pred, average='macro', zero_division=0)
            print(f" → Val Macro-F1: {macro_F1:.4f}")

            if macro_F1 > best_f1:
                best_f1, wait = macro_F1, 0
                yolo.save(os.path.join(project_dir, run_name,'weights','best.pt'))
                torch.save(dgnn.state_dict(), os.path.join(project_dir,run_name,'weights','dgnn_best.pt'))
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping.")
                    break

        energy_consumption = tracker.final_emissions_data.energy_consumed
        co2_emissions = tracker.final_emissions

        avg_time = sum(batch_times) / len(batch_times)
        max_gpu_mem = max(gpu_memories) if gpu_memories else 0
        max_cpu_mem = max(cpu_memories)

        table = [
            ["Epoch", epoch],
            ["Final Loss", f"{running_loss:.4f}"],
            ["Average Batch Time (sec)", f"{avg_time:.4f}"],
            ["Maximum GPU Memory Usage (MB)", f"{max_gpu_mem:.2f}"],
            ["Maximum CPU Memory Usage (MB)", f"{max_cpu_mem:.2f}"],
            ["Energy Consumption (kWh)", f"{energy_consumption:.4f} kWh"],
            ["CO₂ Emissions (kg)", f"{co2_emissions:.4f} kg"],
        ]

        print(tabulate(table, headers=["Metric", "Value"], tablefmt="pretty"))
    
    if torch.cuda.is_available():
        nvmlShutdown()


    return yolo, dgnn, reducer


device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model, dgnn_model, reducer = train_and_validate(
    epochs=50, patience=5, device=device
)

valid_results = run_inference(yolo_model, dgnn_model, reducer, valid_loader,
                            valid_dataset.part_to_idx, valid_dataset.idx_to_part, device, save_vis=True)
part_level_evaluation(valid_results, valid_dataset.part_to_idx, valid_dataset.idx_to_part)

test_results = run_inference(yolo_model, dgnn_model, reducer, test_loader,
                            test_dataset.part_to_idx, test_dataset.idx_to_part, device, save_vis=True)
part_level_evaluation(test_results, test_dataset.part_to_idx, test_dataset.idx_to_part)