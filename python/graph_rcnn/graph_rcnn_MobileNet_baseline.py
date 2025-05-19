import os
import json
import random
from tqdm import tqdm
import time
import psutil
import gc
import numpy as np
from tabulate import tabulate
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torchvision.ops import box_iou
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import nms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch_geometric.nn import GATConv
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlShutdown,
)
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


final_output_json = "/var/scratch/sismail/data/processed/final_annotations_without_occluded.json"
image_directory = "/var/scratch/sismail/data/images"

test_ratio = 0.2
valid_ratio = 0.1
random_seed = 42

with open(final_output_json, "r") as f:
    annotations = json.load(f)

image_filenames = list(annotations["images"].keys())

random.seed(random_seed)
random.shuffle(image_filenames)

num_test = int(len(image_filenames) * test_ratio)
test_images = image_filenames[:num_test]
train_images = image_filenames[num_test:]
num_valid = int(len(train_images) * valid_ratio)
valid_images = train_images[:num_valid]
train_images = train_images[num_valid:]

train_annotations = {
    "all_parts": annotations["all_parts"],
    "images": {img_name: annotations["images"][img_name] for img_name in train_images},
}

valid_annotations = {
    "all_parts": annotations["all_parts"],
    "images": {img_name: annotations["images"][img_name] for img_name in valid_images},
}

test_annotations = {
    "all_parts": annotations["all_parts"],
    "images": {img_name: annotations["images"][img_name] for img_name in test_images},
}


class BikePartsDetectionDataset(Dataset):
    def __init__(
        self,
        annotations_dict,
        image_dir,
        transform=None,
        augment=True,
        target_size=(640, 640),
    ):
        self.all_parts = annotations_dict["all_parts"]
        self.part_to_idx = {part: idx + 1 for idx, part in enumerate(self.all_parts)}
        self.idx_to_part = {idx + 1: part for idx, part in enumerate(self.all_parts)}
        self.image_data = annotations_dict["images"]
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
            image = transforms.functional.adjust_brightness(
                image, brightness_factor=random.uniform(0.6, 1.4)
            )
        if random.random() < 0.8:
            image = transforms.functional.adjust_contrast(
                image, contrast_factor=random.uniform(0.6, 1.4)
            )
        if random.random() < 0.5:
            image = transforms.functional.adjust_saturation(
                image, saturation_factor=random.uniform(0.7, 1.3)
            )

        return image, boxes

    def __getitem__(self, idx):
        real_idx = idx % len(self.image_filenames)
        do_augment = self.augment and (idx >= len(self.image_filenames))

        img_filename = self.image_filenames[real_idx]
        img_path = os.path.join(self.image_dir, img_filename)

        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size

        annotation = self.image_data[img_filename]
        available_parts_info = annotation["available_parts"]
        missing_parts_names = annotation.get("missing_parts", [])

        boxes = []
        labels = []

        for part_info in available_parts_info:
            part_name = part_info["part_name"]
            bbox = part_info["absolute_bounding_box"]
            xmin = bbox["left"]
            ymin = bbox["top"]
            xmax = xmin + bbox["width"]
            ymax = ymin + bbox["height"]
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
            [self.part_to_idx[part] for part in missing_parts_names], dtype=torch.int64
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "missing_labels": missing_labels,
            "image_id": torch.tensor([real_idx]),
        }

        return image, target


train_dataset = BikePartsDetectionDataset(
    annotations_dict=train_annotations, image_dir=image_directory, augment=False
)

valid_dataset = BikePartsDetectionDataset(
    annotations_dict=valid_annotations, image_dir=image_directory, augment=False
)

test_dataset = BikePartsDetectionDataset(
    annotations_dict=test_annotations, image_dir=image_directory, augment=False
)

train_loader = DataLoader(
    train_dataset,
    worker_init_fn=seed_worker,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch)),
)


def evaluate_model(model, loader, part_to_idx, device):
    model.eval()
    all_parts = set(part_to_idx.values())
    results = []
    for images, targets in tqdm(loader):
        images = [i.to(device) for i in images]
        preds = model(images)
        for i,(p,t) in enumerate(zip(preds,targets)):
            pred_set = set(p['labels'].cpu().tolist())
            true_set = set(t['missing_labels'].tolist())
            results.append({'predicted_missing_parts':all_parts-pred_set,'true_missing_parts':true_set})
    return results

def part_level_evaluation(results, part_to_idx, idx_to_part):
    parts = list(part_to_idx.values())
    Yt = np.array([[1 if p in r['true_missing_parts'] else 0 for p in parts] for r in results])
    Yp = np.array([[1 if p in r['predicted_missing_parts'] else 0 for p in parts] for r in results])
    micro = f1_score(Yt,Yp,average='micro',zero_division=0)
    macro = f1_score(Yt,Yp,average='macro',zero_division=0)
    fn = np.logical_and(Yt==1,Yp==0).sum()
    tp = np.logical_and(Yt==1,Yp==1).sum()
    fp = np.logical_and(Yt==0,Yp==1).sum()
    miss_rate = fn/(fn+tp) if fn+tp>0 else 0
    fppi = fp/len(results)
    acc = accuracy_score(Yt.flatten(),Yp.flatten())
    prec = precision_score(Yt.flatten(),Yp.flatten(),zero_division=0)
    rec = recall_score(Yt.flatten(),Yp.flatten(),zero_division=0)
    f1 = f1_score(Yt.flatten(),Yp.flatten(),zero_division=0)
    print(f"Micro-F1: {micro:.4f}  Macro-F1: {macro:.4f}  MissRate: {miss_rate:.4f}  FPPI: {fppi:.4f}")
    print(f"Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")
    table=[]
    for j,p in enumerate(parts):
        a = accuracy_score(Yt[:,j],Yp[:,j])
        p_ = precision_score(Yt[:,j],Yp[:,j],zero_division=0)
        r_ = recall_score(Yt[:,j],Yp[:,j],zero_division=0)
        f_ = f1_score(Yt[:,j],Yp[:,j],zero_division=0)
        table.append([idx_to_part[p],f"{a:.3f}",f"{p_:.3f}",f"{r_:.3f}",f"{f_:.3f}"])
    print(tabulate(table,headers=["Part","Acc","Prec","Rec","F1"],tablefmt="fancy_grid"))

detector = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
in_feats = detector.roi_heads.box_predictor.cls_score.in_features

detector.roi_heads.box_predictor = FastRCNNPredictor(
    in_feats, len(train_dataset.all_parts) + 1
)


class RelationProposalNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim * 2 + 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, features, boxes):
        N, _ = features.size()
        f1 = features.unsqueeze(1).expand(-1, N, -1)
        f2 = features.unsqueeze(0).expand(N, -1, -1)
        b1 = boxes.unsqueeze(1).expand(-1, N, -1)
        b2 = boxes.unsqueeze(0).expand(N, -1, -1)
        geom = torch.abs(b1 - b2)
        x = nn.functional.relu(self.fc1(torch.cat([f1, f2, geom], -1)))
        return self.fc2(x).squeeze(-1)


class AttentionalGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4):
        super().__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.layer2 = GATConv(hidden_dim * heads, out_dim, heads=1)

    def forward(self, x, edge_index):
        x = nn.functional.relu(self.layer1(x, edge_index))
        return self.layer2(x, edge_index)


class GraphRCNN(nn.Module):
    def __init__(self, detector, num_parts, topk=3, gcn_hidden=256):
        super().__init__()
        self.backbone = detector.backbone
        self.rpn = detector.rpn
        self.roi_pool = detector.roi_heads.box_roi_pool
        self.box_head = detector.roi_heads.box_head
        self.transform = detector.transform
        feat_dim = self.box_head.fc7.out_features
        self.relation = RelationProposalNetwork(feat_dim)

        self.gcn = AttentionalGCN(feat_dim + 4, gcn_hidden, num_parts + 1)
        self.topk = topk

    def forward(self, images, targets=None):
        imgs, img_sizes = self.transform(images)
        features = self.backbone(imgs.tensors)
        proposals, losses_rpn = self.rpn(imgs, features, targets)
        pooled = self.roi_pool(features, proposals, [img.shape[-2:] for img in images])
        node_feats = self.box_head(pooled)
        if self.training:
            gt_boxes = torch.cat([t["boxes"] for t in targets], 0)
            gt_labels = torch.cat([t["labels"] for t in targets], 0)
            pooled_gt = self.roi_pool(features, [gt_boxes], [images[0].shape[-2:]])
            feats_gt = self.box_head(pooled_gt).squeeze(0)
            rel = self.relation(feats_gt, gt_boxes)
            idx = torch.topk(rel, self.topk + 1, dim=1).indices[:, 1:]
            src = idx.flatten()
            dst = (
                torch.arange(rel.size(0), device=rel.device)
                .unsqueeze(1)
                .expand(-1, self.topk)
                .flatten()
            )
            edge = torch.stack([dst, src], 0)
            h, w = images[0].shape[-2], images[0].shape[-1]
            nb = gt_boxes.clone()
            nb[:, [0, 2]] /= w
            nb[:, [1, 3]] /= h
            geom = torch.stack(
                [nb[:, 0], nb[:, 1], nb[:, 2] - nb[:, 0], nb[:, 3] - nb[:, 1]], 1
            )
            gcn_in = torch.cat([feats_gt, geom], 1)
            logits = self.gcn(gcn_in, edge)
            loss_gcn = nn.functional.cross_entropy(logits, gt_labels)
            total_loss = sum(losses_rpn.values()) + loss_gcn
            losses_rpn["loss_gcn"] = loss_gcn
            return total_loss, losses_rpn
        else:
            props = torch.cat(proposals, 0)
            feats = node_feats
            rel = self.relation(feats, props)
            idx = torch.topk(rel, self.topk + 1, dim=1).indices[:, 1:]
            src = idx.flatten()
            dst = (
                torch.arange(rel.size(0), device=rel.device)
                .unsqueeze(1)
                .expand(-1, self.topk)
                .flatten()
            )
            edge = torch.stack([dst, src], 0)
            h, w = images[0].shape[-2], images[0].shape[-1]
            nb = props.clone()
            nb[:, [0, 2]] /= w
            nb[:, [1, 3]] /= h
            geom = torch.stack(
                [nb[:, 0], nb[:, 1], nb[:, 2] - nb[:, 0], nb[:, 3] - nb[:, 1]], 1
            )
            gcn_in = torch.cat([feats, geom], 1)
            logits = self.gcn(gcn_in, edge)
            labels = logits.argmax(1)
            return [{"boxes": props, "labels": labels}]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GraphRCNN(detector, len(train_dataset.all_parts)).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

if torch.cuda.is_available():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

epochs = 1
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
                images = [image.to(device) for image in images]
                targets = [{'boxes':t['boxes'].to(device),'labels':t['labels'].to(device),'image_id':t['image_id'].to(device)} for t in targets]

                start_time = time.time()

                optimizer.zero_grad()
                total_loss, loss_dict = model(images, targets)
                total_loss.backward()
                optimizer.step()

                end_time = time.time()
                inference_time = end_time - start_time
                batch_times.append(inference_time)

                if torch.cuda.is_available():
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    gpu_mem_used = mem_info.used / (1024**2)
                    gpu_memories.append(gpu_mem_used)
                else:
                    gpu_mem_used = 0

                cpu_mem_used = psutil.virtual_memory().used / (1024**2)
                cpu_memories.append(cpu_mem_used)

                tepoch.set_postfix(
                    {
                        "loss": f"{total_loss.item():.4f}",
                        "time (s)": f"{inference_time:.3f}",
                        "GPU Mem (MB)": f"{gpu_mem_used:.0f}",
                        "CPU Mem (MB)": f"{cpu_mem_used:.0f}",
                    }
                )

                del loss_dict, images, targets
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
                torch.save(
                    model.state_dict(),
                    "/var/scratch/sismail/models/graph_rcnn/graphrcnn_MobileNet_baseline_model.pth",
                )
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
        ["Final Loss", f"{total_loss.item():.4f}"],
        ["Average Batch Time (sec)", f"{avg_time:.4f}"],
        ["Maximum GPU Memory Usage (MB)", f"{max_gpu_mem:.2f}"],
        ["Maximum CPU Memory Usage (MB)", f"{max_cpu_mem:.2f}"],
        ["Energy Consumption (kWh)", f"{energy_consumption:.4f} kWh"],
        ["CO₂ Emissions (kg)", f"{co2_emissions:.4f} kg"],
    ]

    print(tabulate(table, headers=["Metric", "Value"], tablefmt="pretty"))


model.load_state_dict(
    torch.load(
        "/var/scratch/sismail/models/graph_rcnn/graphrcnn_MobileNet_baseline_model.pth", map_location=device
    )
)
model.to(device)

model.eval()

results_per_image = evaluate_model(
    model, valid_loader, train_dataset.part_to_idx, device
)

part_level_evaluation(
    results_per_image, train_dataset.part_to_idx, train_dataset.idx_to_part
)


results_per_image = evaluate_model(
    model, test_loader, train_dataset.part_to_idx, device
)

part_level_evaluation(
    results_per_image, train_dataset.part_to_idx, train_dataset.idx_to_part
)
