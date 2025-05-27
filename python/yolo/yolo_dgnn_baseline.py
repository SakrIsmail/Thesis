import os
import shutil
import json
import random
from tqdm import tqdm
import logging
import time
import psutil
import gc
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

class YOLOv8Wrapper(nn.Module):
    def __init__(self, model_path='yolov8n.pt'):
        super().__init__()
        self.model = YOLO(model_path)
        self.model.model[8].register_forward_hook(self.hook_fn)
        self._features = []

    def hook_fn(self, module, input, output):
        # This assumes output is [B, C, H, W]
        self._features.append(output)

    def forward(self, images, targets=None):
        self._features.clear()
        results = self.model(images, verbose=False)

        loss = None
        if self.model.training and targets is not None:
            loss = self.model.loss_func(self.model, results, targets)[0]

        # Extract spatial features and pool to (N_detections, feat_dim)
        pooled_feats = []
        for i, det in enumerate(results):
            boxes = det.boxes.xyxy  # in (x1, y1, x2, y2) format
            fmap = self._features[0][i]  # [C, H, W]
            if boxes.numel() == 0:
                pooled_feats.append(torch.zeros((0, fmap.shape[0]), device=fmap.device))
                continue
            H, W = fmap.shape[1:]
            feat_per_box = []
            for box in boxes:
                x1, y1, x2, y2 = box
                x1 = int((x1 / 640) * W)
                x2 = int((x2 / 640) * W)
                y1 = int((y1 / 640) * H)
                y2 = int((y2 / 640) * H)
                region = fmap[:, y1:y2, x1:x2]
                if region.numel() == 0:
                    pooled = torch.zeros((fmap.shape[0],), device=fmap.device)
                else:
                    pooled = torch.mean(region.view(fmap.shape[0], -1), dim=1)
                feat_per_box.append(pooled)
            pooled_feats.append(torch.stack(feat_per_box))

        return results, pooled_feats, loss

    

def construct_graph_inputs(stored_feats, predictions, device):
    sigma_spatial = 20.0
    gamma_appear = 5.0
    alpha = 1.0 / (2 * sigma_spatial**2)
    weight_threshold = 1e-3

    graph_data = []

    for feats, det in zip(stored_feats, predictions):
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
        sim_feats = torch.nn.functional.cosine_similarity(feats.unsqueeze(1), feats.unsqueeze(0), dim=2)
        w_appear = torch.exp(gamma_appear * sim_feats)
        W = w_spatial * w_appear
        src, dst = (W > weight_threshold).nonzero(as_tuple=True)
        if src.numel() == 0: continue
        edge_index = torch.stack([src, dst], dim=0)
        edge_weight = W[src, dst]
        graph_data.append((x, edge_index, edge_weight))

    return graph_data


class CombinedModel(nn.Module):
    def __init__(self, yolo_model_path='yolov8n.pt', feat_dim=256, hidden_dim=512):
        super().__init__()
        self.yolo_wrapper = YOLOv8Wrapper(yolo_model_path)
        self.dgnn = SpatialDGNN(feat_dim=feat_dim, hidden_dim=hidden_dim)

    def forward(self, images, targets=None, device='cuda'):
        yolo_results, features = self.yolo_wrapper(images, targets)

        with torch.no_grad():
            graph_data = construct_graph_inputs(features, yolo_results, device)
        total_gnn_loss = 0.0

        for x, edge_index, edge_weight in graph_data:
            refined_features = self.dgnn(x, edge_index, edge_weight)
            total_gnn_loss += nn.functional.mse_loss(refined_features, x)

        gnn_loss = total_gnn_loss / len(graph_data) if graph_data else torch.tensor(0.0, device=device)


        yolo_results, features, yolo_loss = self.yolo_wrapper(images, targets)

        total_loss = yolo_loss + 0.1 * gnn_loss
        return total_loss, yolo_results



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

def convert_to_yolo_format(targets, image_size=(640, 640)):
    yolo_targets = []
    img_w, img_h = image_size

    for target in targets:
        boxes = target['boxes']
        labels = target['labels']

        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        cx /= img_w
        cy /= img_h
        w /= img_w
        h /= img_h

        norm_boxes = torch.stack([cx, cy, w, h], dim=1)

        yolo_targets.append({
            "cls": labels,
            "bboxes": norm_boxes
        })

    return yolo_targets

def evaluate_model(model, dataloader, part_to_idx, device):
    model.eval()
    results = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images = [img.to(device) for img in images]
            true_missing = [t["missing_labels"].tolist() for t in targets]

            yolo_results, stored_feats, _ = model.yolo_wrapper(images)
            graph_inputs = construct_graph_inputs(stored_feats, yolo_results, device)
            predicted_missing_parts = []

            for (x, edge_index, edge_weight), det in zip(graph_inputs, yolo_results):
                refined_feats = model.dgnn(x, edge_index, edge_weight)
                part_logits = refined_feats[:, -len(part_to_idx):]
                part_probs = torch.sigmoid(part_logits).mean(dim=0)
                pred_parts = (part_probs < 0.5).nonzero(as_tuple=True)[0].tolist()
                predicted_missing_parts.append(pred_parts)

            for i in range(len(images)):
                results.append({
                    "true_missing_parts": true_missing[i],
                    "predicted_missing_parts": predicted_missing_parts[i]
                })

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CombinedModel(yolo_model_path='yolov8n.pt').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)


if torch.cuda.is_available():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

epochs = 50
patience = 5
best_macro_f1 = 0
no_improve = 0

for epoch in range(1, epochs + 1):
    with EmissionsTracker(log_level="critical", save_to_file=False) as tracker:
        model.train()

        batch_times = []
        gpu_memories = []
        cpu_memories = []

        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch}/{epochs}") as tepoch:
            for images, targets in tepoch:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                yolo_targets = convert_to_yolo_format(targets)

                start_time = time.time()

                optimizer.zero_grad()
                loss, _ = model(images, yolo_targets, device=device)
                loss.backward()
                optimizer.step()

                end_time = time.time()
                inference_time = end_time - start_time
                batch_times.append(inference_time)

                if torch.cuda.is_available():
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    gpu_mem_used = mem_info.used / (1024 ** 2)
                    gpu_memories.append(gpu_mem_used)
                else:
                    gpu_mem_used = 0

                cpu_mem_used = psutil.virtual_memory().used / (1024 ** 2)
                cpu_memories.append(cpu_mem_used)

                tepoch.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "time (s)": f"{inference_time:.3f}",
                    "GPU Mem (MB)": f"{gpu_mem_used:.0f}",
                    "CPU Mem (MB)": f"{cpu_mem_used:.0f}"
                })

                del loss, images, targets, yolo_targets
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        model.eval()
        results = evaluate_model(model, valid_loader, train_dataset.part_to_idx, device)
        parts = list(train_dataset.part_to_idx.values())
        Y_true = np.array([[1 if p in r['true_missing_parts'] else 0 for p in parts] for r in results])
        Y_pred = np.array([[1 if p in r['predicted_missing_parts'] else 0 for p in parts] for r in results])
        macro_f1 = f1_score(Y_true, Y_pred, average='macro', zero_division=0)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            no_improve = 0
            torch.save(model.state_dict(), "/var/scratch/sismail/models/yolo_gnn_baseline_model.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break


    energy_consumption = tracker.final_emissions_data.energy_consumed
    co2_emissions = tracker.final_emissions

    avg_time = sum(batch_times) / len(batch_times)
    max_gpu_mem = max(gpu_memories) if gpu_memories else 0
    max_cpu_mem = max(cpu_memories)

    table = [
        ["Epoch", epoch],
        ["Best Macro F1", f"{best_macro_f1:.4f}"],
        ["Average Batch Time (sec)", f"{avg_time:.4f}"],
        ["Maximum GPU Memory Usage (MB)", f"{max_gpu_mem:.2f}"],
        ["Maximum CPU Memory Usage (MB)", f"{max_cpu_mem:.2f}"],
        ["Energy Consumption (kWh)", f"{energy_consumption:.4f}"],
        ["CO₂ Emissions (kg)", f"{co2_emissions:.4f}"],
    ]

    print(tabulate(table, headers=["Metric", "Value"], tablefmt="pretty"))

if torch.cuda.is_available():
    nvmlShutdown()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CombinedModel(yolo_model_path='yolov8n.pt').to(device)
model.load_state_dict(torch.load("/var/scratch/sismail/models/yolo_gnn_baseline_model.pth"))


val_results = evaluate_model(model, valid_loader, valid_dataset.part_to_idx, device)
test_results = evaluate_model(model, valid_loader, valid_dataset.part_to_idx, device)


part_level_evaluation(val_results,  valid_dataset.part_to_idx,  valid_dataset.idx_to_part)
part_level_evaluation(test_results, test_dataset.part_to_idx, test_dataset.idx_to_part)