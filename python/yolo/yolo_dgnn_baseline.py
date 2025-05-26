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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

stored_feats = []
dgnn = None
gnn_opt = None

batch_times, gpu_memories, cpu_memories = [], [], []
batch_count = 0
nvml_handle, em_tracker = None, None
start_time = 0
best_macro_f1 = 0.0
no_improve_epochs = 0
patience = 5


def on_train_start(trainer):
    global dgnn, gnn_opt, stored_feats
    head = trainer.model.model[-2]
    head.register_forward_hook(lambda m, inp, out: stored_feats.append(out))
    dgnn = SpatialDGNN().to(trainer.device)
    gnn_opt = torch.optim.AdamW(dgnn.parameters(), lr=1e-4)
    stored_feats.clear()


def on_train_epoch_start(trainer):
    global batch_times, gpu_memories, cpu_memories, nvml_handle, em_tracker, batch_count
    batch_times.clear()
    gpu_memories.clear()
    cpu_memories.clear()
    batch_count = 0
    em_tracker = EmissionsTracker(log_level="critical", save_to_file=False)
    em_tracker.__enter__()
    if trainer.device.type == 'cuda':
        nvmlInit()
        nvml_handle = nvmlDeviceGetHandleByIndex(0)

def on_train_batch_start(trainer):
    global start_time, stored_feats
    start_time = time.time()
    stored_feats.clear()  


def on_before_zero_grad(trainer):
    preds = trainer.predictions
    total_gnn_loss = 0.0
    device = trainer.device

    sigma_spatial = 20.0
    gamma_appear = 5.0
    alpha = 1.0 / (2 * sigma_spatial**2)
    weight_threshold = 1e-3

    for feats, det in zip(stored_feats, preds):
        Ni = feats.size(0)
        if Ni < 2: continue

        boxes = det.boxes.xyxy.to(device)
        confs = det.boxes.conf.to(device).unsqueeze(1)
        clss  = det.boxes.cls.to(device).unsqueeze(1)
        feats = feats.to(device)

        cxs = ((boxes[:,0] + boxes[:,2]) / 2).unsqueeze(1)
        cys = ((boxes[:,1] + boxes[:,3]) / 2).unsqueeze(1)
        ws  = (boxes[:,2] - boxes[:,0]).unsqueeze(1)
        hs  = (boxes[:,3] - boxes[:,1]).unsqueeze(1)
        x = torch.cat([cxs, cys, ws, hs, confs, clss, feats], dim=1)

        centers = torch.cat([cxs, cys], dim=1)
        dist_mat = torch.cdist(centers, centers, p=2)
        w_spatial = torch.exp(-alpha * dist_mat**2)
        sim_feats = nn.functional.cosine_similarity(feats.unsqueeze(1), feats.unsqueeze(0), dim=2)
        w_appear = torch.exp(gamma_appear * sim_feats)
        W = w_spatial * w_appear
        src, dst = (W > weight_threshold).nonzero(as_tuple=True)
        if src.numel() == 0:
            continue
        edge_index = torch.stack([src, dst], dim=0)
        edge_weight = W[src, dst]
        x_ref = dgnn(x, edge_index, edge_weight)
        total_gnn_loss += nn.functional.mse_loss(x_ref[:, -256:], x[:, -256:])
    if len(preds) > 0:
        total_gnn_loss = total_gnn_loss / len(preds)
        lmbda = 0.1
        combined = trainer.loss + lmbda * total_gnn_loss
        trainer.optimizer.zero_grad()
        gnn_opt.zero_grad()
        combined.backward()
        gnn_opt.step()
        trainer.loss = combined.item()
    
def optimizer_step(trainer):
    trainer.optimizer.step()

def on_train_batch_end(trainer):
    global batch_count
    batch_count += 1
    end_time = time.time()
    inference_time = end_time - start_time
    batch_times.append(inference_time)
    if nvml_handle:
        mi = nvmlDeviceGetMemoryInfo(nvml_handle)
        gpu_memories.append(mi.used / 1024**2)
    else:
        gpu_memories.append(0)
    cpu_memories.append(psutil.virtual_memory().used / 1024**2)
    print(f"Batch {batch_count} | loss={trainer.loss:.4f} | time={inference_time:.3f}s | "
          f"GPU={gpu_memories[-1]:.0f}MB | CPU={cpu_memories[-1]:.0f}MB", file=sys.stderr)
    gc.collect()


def on_train_epoch_end(trainer):
    global nvml_handle, em_tracker

    em_tracker.__exit__(None, None, None)
    energy = em_tracker.final_emissions_data.energy_consumed
    co2 = em_tracker.final_emissions

    if nvml_handle:
        nvmlShutdown()
    table = [
        ['Epoch', trainer.epoch],
        ['Final Loss', f"{trainer.loss:.4f}"],
        ['Avg Batch Time (s)', f"{np.mean(batch_times):.4f}"],
        ['Max GPU Mem (MB)', f"{np.max(gpu_memories):.1f}"],
        ['Max CPU Mem (MB)', f"{np.max(cpu_memories):.1f}"],
        ['Energy (kWh)', f"{energy:.4f}"],
        ['CO₂ (kg)', f"{co2:.4f}"],
    ]
    print(tabulate(table, headers=["Metric","Value"], tablefmt="pretty"))

def on_fit_epoch_end(trainer, wrapper):
    global best_macro_f1, no_improve_epochs, patience, valid_loader, device
    trainer.save_model()
    wdir = os.path.join(trainer.args.project, trainer.args.name, 'weights')
    last_yolo = os.path.join(wdir, 'last.pt')
    last_dgnn = os.path.join(wdir, 'dgnn_last.pt')
    torch.save(dgnn.state_dict(), last_dgnn)

    val_model = YOLO(last_yolo)
    val_model.to(trainer.device).eval()
    dgnn_eval = SpatialDGNN().to(trainer.device)
    dgnn_eval.load_state_dict(torch.load(last_dgnn, map_location=trainer.device))
    dgnn_eval.eval()


    results = wrapper.run_inference(valid_loader, valid_dataset.part_to_idx, valid_dataset.idx_to_part, early_stopping=True)

    parts = list(valid_dataset.part_to_idx.values())
    Y_true = np.array([[1 if p in r['true_missing_parts'] else 0 for p in parts] for r in results])
    Y_pred = np.array([[1 if p in r['predicted_missing_parts'] else 0 for p in parts] for r in results])
    macro_f1 = f1_score(Y_true, Y_pred, average='macro', zero_division=0)

    print(f"Epoch {trainer.epoch + 1}: Macro F1 Score = {macro_f1:.4f}")

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        no_improve_epochs = 0
        shutil.copy(last_yolo, os.path.join(wdir, 'best.pt'))
        shutil.copy(last_dgnn, os.path.join(wdir, 'dgnn_best.pt'))
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {trainer.epoch + 1}")
            trainer.stop_training = True


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


class BikePartsTrainer:
    def __init__(self, device, model_path='yolov8m.pt'):
        self.device = device
        
        self.model = YOLO(model_path, verbose=False)
        
        self.dgnn = SpatialDGNN().to(device)
        self.feat_reducer = nn.Linear(576, 256).to(self.device)
        
        self.callbacks = []
        
    def add_callbacks(self, callbacks):
        """Add callbacks for training."""
        for event_name, func in callbacks:
            self.model.add_callback(event_name, func)
        self.callbacks.extend(callbacks)
        
    def train(self, data_yaml, epochs=50, batch=16, imgsz=640, project_path=None, run_name=None):
        """Run training with callbacks and saving."""
        
        self.model.to(self.device)
        
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            optimizer='AdamW',
            lr0=1e-4,
            weight_decay=1e-4,
            workers=4,
            device=self.device,
            seed=42,
            verbose=False,
            plots=False,
            project=project_path,
            name=run_name,
            exist_ok=True,
            save=True
        )
        
        # After training, save DGNN weights manually if needed
        dgnn_path = os.path.join(project_path, run_name, 'weights', 'dgnn_best.pt')
        torch.save(self.dgnn.state_dict(), dgnn_path)
        print(f"Saved DGNN weights to {dgnn_path}")
        
    def load_trained_models(self, yolo_weights_path, dgnn_weights_path):
        """Load YOLO and DGNN weights for inference."""
        self.model = YOLO(yolo_weights_path).to(self.device).eval()
        self.dgnn.load_state_dict(torch.load(dgnn_weights_path, map_location=self.device))
        self.dgnn.eval()
        
    def run_inference(self, loader, part_to_idx, idx_to_part,
                  sigma_spatial=20.0, gamma_appear=5.0,
                  weight_threshold=1e-3, conf_threshold=0.5,
                  early_stopping=False):
        """Run combined YOLO + DGNN inference, no callbacks."""
        
        stored_feats = []
        detect_layer = self.model.model.model[-2]
        
        hook_handle = detect_layer.register_forward_hook(
            lambda m, inp, out: stored_feats.append(out)
        )
        
        alpha = 1.0 / (2 * sigma_spatial**2)
        results = []
        num_imgs = 0
        
        stride = 32  # Adjust this if your feature map stride differs
        with torch.no_grad():
            for images, targets in tqdm(loader, desc="DGNN Inference"):
                stored_feats.clear()
                np_images = [
                    (img.cpu().permute(1,2,0).numpy() * 255).clip(0,255).astype('uint8')
                    for img in images
                ]
                preds = self.model(np_images, device=self.device, verbose=False)
                
                for i, (feat_tensor, det, target) in enumerate(zip(stored_feats, preds, targets)):
                    feats = feat_tensor.to(self.device)  # shape [1, C, H, W]
                    
                    boxes = det.boxes.xyxy.to(self.device)  # [num_boxes, 4]
                    confs = det.boxes.conf.to(self.device).unsqueeze(1)  # [num_boxes, 1]
                    clss  = det.boxes.cls.to(self.device).unsqueeze(1)  # [num_boxes, 1]
                    cxs = ((boxes[:,0] + boxes[:,2]) / 2).unsqueeze(1)  # [num_boxes, 1]
                    cys = ((boxes[:,1] + boxes[:,3]) / 2).unsqueeze(1)  # [num_boxes, 1]
                    ws  = (boxes[:,2] - boxes[:,0]).unsqueeze(1)        # [num_boxes, 1]
                    hs  = (boxes[:,3] - boxes[:,1]).unsqueeze(1)        # [num_boxes, 1]
                    
                    num_boxes = boxes.shape[0]
                    if num_boxes > 0:
                        # Scale boxes for roi_align (x1,y1,x2,y2) / stride
                        boxes_scaled = boxes / stride
                        
                        # Batch indices for roi_align: all zeros since one image at a time
                        batch_indices = torch.zeros(num_boxes, 1, device=self.device)
                        
                        # Boxes for roi_align [num_boxes, 5]: (batch_idx, x1, y1, x2, y2)
                        boxes_for_roi = torch.cat([batch_indices, boxes_scaled], dim=1)
                        
                        # ROI align to get pooled features of size 1x1 per box
                        pooled_feats = roi_align(feats, boxes_for_roi, output_size=(1, 1))
                        
                        # Flatten pooled features to [num_boxes, C]
                        pooled_feats = pooled_feats.view(num_boxes, -1)
                        
                        pooled_feats_reduced = self.feat_reducer(pooled_feats)
                        x = torch.cat([cxs, cys, ws, hs, confs, clss, pooled_feats_reduced], dim=1)
                    else:
                        # No detections: create empty tensor with proper feature size
                        x = torch.empty((0, 6 + feats.shape[1]), device=self.device)
                    
                    centers = torch.cat([cxs, cys], dim=1)
                    dist_mat = torch.cdist(centers, centers, p=2)
                    w_spatial = torch.exp(-alpha * dist_mat**2)
                    sim_feats = nn.functional.cosine_similarity(
                        pooled_feats.unsqueeze(1), pooled_feats.unsqueeze(0), dim=2) if num_boxes > 0 else torch.tensor([])
                    w_appear = torch.exp(gamma_appear * sim_feats) if num_boxes > 0 else torch.tensor([])
                    
                    if num_boxes > 0:
                        W = w_spatial * w_appear
                        
                        src, dst = (W > weight_threshold).nonzero(as_tuple=True)
                        if src.numel() > 0:
                            edge_index  = torch.stack([src, dst], dim=0)
                            edge_weight = W[src, dst]
                            with torch.no_grad():
                                x_ref = self.dgnn(x, edge_index, edge_weight)
                            refined_confs = x_ref[:, 4].cpu()
                            refined_cls   = x_ref[:, 5].round().clamp(0, len(part_to_idx)-1).long().cpu()
                        else:
                            refined_confs = confs.squeeze(1).cpu()
                            refined_cls   = clss.squeeze(1).cpu().long()
                    else:
                        refined_confs = torch.tensor([], device='cpu')
                        refined_cls = torch.tensor([], dtype=torch.long, device='cpu')
                    
                    # Optional visualization for first few images
                    if num_imgs < 3 and not early_stopping:
                        filename = f"img{target['image_id'].item():04d}.png"
                        save_path = os.path.join('/home/$USER/Thesis/visualisations', filename)
                        save_detection_graph(
                            np_images[i],
                            det.boxes.xyxy,
                            W if num_boxes > 0 else None,
                            edge_index if num_boxes > 0 and src.numel() > 0 else None,
                            output_path=save_path,
                            title=f"Image {target['image_id'].item()}"
                        )
                        num_imgs += 1
                    
                    keep = (refined_confs >= conf_threshold).nonzero().squeeze(1)
                    pred_labels = set(refined_cls[keep].tolist()) if keep.numel() > 0 else set()
                    true_missing = set(target['missing_labels'].tolist())
                    all_parts = set(part_to_idx.values())
                    
                    results.append({
                        'image_id': target['image_id'].item(),
                        'predicted_missing_parts': all_parts - pred_labels,
                        'true_missing_parts': true_missing
                    })
            
            hook_handle.remove()
        return results



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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = BikePartsTrainer(device)

trainer.add_callbacks([
    ('on_train_start', on_train_start),
    ('on_train_epoch_start', on_train_epoch_start),
    ('on_train_batch_start', on_train_batch_start),
    ('on_before_zero_grad', on_before_zero_grad),
    ('optimizer_step', optimizer_step),
    ('on_train_batch_end', on_train_batch_end),
    ('on_train_epoch_end', on_train_epoch_end),
    ('on_fit_epoch_end', partial(on_fit_epoch_end, wrapper=trainer)),
])


trainer.train(
    data_yaml='/var/scratch/sismail/data/yolo_format/noaug/data.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    project_path='/var/scratch/sismail/models/yolo/runs',
    run_name='bikeparts_dgnn_euclid'
)

trainer.load_trained_models(
    yolo_weights_path='/var/scratch/sismail/models/yolo/runs/bikeparts_dgnn_euclid/weights/best.pt',
    dgnn_weights_path='/var/scratch/sismail/models/yolo/runs/bikeparts_dgnn_euclid/weights/dgnn_best.pt'
)

val_results = trainer.run_inference(
    loader=valid_loader,
    part_to_idx=valid_dataset.part_to_idx,
    idx_to_part=valid_dataset.idx_to_part
)

# Run inference on test set
test_results = trainer.run_inference(
    loader=test_loader,
    part_to_idx=test_dataset.part_to_idx,
    idx_to_part=test_dataset.idx_to_part
)

part_level_evaluation(val_results,  valid_dataset.part_to_idx,  valid_dataset.idx_to_part)
part_level_evaluation(test_results, test_dataset.part_to_idx, test_dataset.idx_to_part)