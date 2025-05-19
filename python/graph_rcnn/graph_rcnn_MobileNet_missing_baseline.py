import os
import json
import random
from tqdm import tqdm
import time
import psutil
import gc
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
from codecarbon import EmissionsTracker
from torchvision.ops import box_iou
from torchvision.ops import nms
from torch_geometric.nn import GATConv


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


final_output_json = "/var/scratch/sismail/data/processed/final_direct_missing.json"
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
        self, annotations_dict, image_dir, augment=True, target_size=(640, 640)
    ):
        self.all_parts = annotations_dict["all_parts"]
        self.part_to_idx = {
            p: i + 1 for i, p in enumerate(self.all_parts)
        }  # background=0
        self.image_data = annotations_dict["images"]
        self.image_filenames = list(self.image_data.keys())
        self.image_dir = image_dir
        self.augment = augment
        self.target_size = target_size

    def __len__(self):
        return len(self.image_filenames) * (2 if self.augment else 1)

    def apply_augmentation(self, image, boxes):
        # horizontal flip
        if random.random() < 0.5:
            image = transforms.functional.hflip(image)
            w = image.width
            boxes = boxes.clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        # color jitters
        if random.random() < 0.8:
            image = transforms.functional.adjust_brightness(
                image, random.uniform(0.6, 1.4)
            )
        if random.random() < 0.8:
            image = transforms.functional.adjust_contrast(
                image, random.uniform(0.6, 1.4)
            )
        if random.random() < 0.5:
            image = transforms.functional.adjust_saturation(
                image, random.uniform(0.7, 1.3)
            )
        return image, boxes

    def __getitem__(self, idx):
        real_idx = idx % len(self.image_filenames)
        do_aug = self.augment and (idx >= len(self.image_filenames))

        fn = self.image_filenames[real_idx]
        img = Image.open(os.path.join(self.image_dir, fn)).convert("RGB")
        ow, oh = img.size

        parts = self.image_data[fn]["parts"]
        boxes, labels, is_missing = [], [], []
        for part in parts:
            bb = part["absolute_bounding_box"]
            x0, y0 = bb["left"], bb["top"]
            x1, y1 = x0 + bb["width"], y0 + bb["height"]
            boxes.append([x0, y0, x1, y1])
            idx = self.part_to_idx[part["part_name"]]
            labels.append(idx)
            is_missing.append(0 if part["present"] else 1)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        is_missing = torch.tensor(is_missing, dtype=torch.int64)

        if do_aug:
            img, boxes = self.apply_augmentation(img, boxes)

        img = transforms.functional.resize(img, self.target_size)
        sx, sy = self.target_size[0] / ow, self.target_size[1] / oh
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        img = transforms.functional.to_tensor(img)

        return img, {"boxes": boxes, "labels": labels, "is_missing": is_missing}


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
    num_workers=0,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    collate_fn=lambda batch: tuple(zip(*batch)),
)


def visualize_and_save_predictions(
    model, dataset, device, out_dir="output_preds", n_images=5
):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    for idx in range(min(n_images, len(dataset))):
        img, target = dataset[idx]
        with torch.no_grad():
            pred = model([img.to(device)])[0]

        img_np = img.mul(255).permute(1, 2, 0).byte().cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_np)

        gt_boxes = target["boxes"][target["is_missing"] == 1].cpu().numpy()
        for x0, y0, x1, y1 in gt_boxes:
            rect = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)

        pred_boxes = pred["boxes_missing"].cpu().numpy()
        for x0, y0, x1, y1 in pred_boxes:
            rect = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="b", facecolor="none"
            )
            ax.add_patch(rect)

        ax.axis("off")
        plt.savefig(
            os.path.join(out_dir, f"graph_pred_{idx}.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)


def evaluate_model(model, loader, all_parts, device, iou_thr=0.5):
    model.eval()
    P = len(all_parts)
    results = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outs = model(images)

            for tgt, out in zip(targets, outs):
                true_vec = torch.zeros(P, dtype=torch.int64)
                for lbl, miss in zip(tgt["labels"], tgt["is_missing"]):
                    if miss.item() == 1:
                        true_vec[lbl.item() - 1] = 1

                pred_vec = torch.zeros(P, dtype=torch.int64)
                boxes_p  = out["boxes_missing"]
                labels_p = out["labels_missing"]
                for b_pred, lbl_pred in zip(boxes_p, labels_p):
                    part_idx = lbl_pred.item() - 1
                    mask = (tgt["labels"] == lbl_pred)
                    if not mask.any():
                        continue
                    gt_boxes = tgt["boxes"][mask]
                    ious = box_iou(b_pred.unsqueeze(0), gt_boxes)
                    if ious.max() >= iou_thr:
                        pred_vec[part_idx] = 1

                results.append({
                    "true_missing_parts":      true_vec.cpu().numpy(),
                    "predicted_missing_parts": pred_vec.cpu().numpy(),
                })
    return results


def part_level_evaluation(results, part_to_idx, idx_to_part):
    parts = list(part_to_idx.values())

    Y_true = np.array(
        [[1 if p in r["true_missing_parts"] else 0 for p in parts] for r in results]
    )
    Y_pred = np.array(
        [
            [1 if p in r["predicted_missing_parts"] else 0 for p in parts]
            for r in results
        ]
    )

    micro_f1 = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(Y_true, Y_pred, average="macro", zero_division=0)

    FN = np.logical_and(Y_true == 1, Y_pred == 0).sum()
    TP = np.logical_and(Y_true == 1, Y_pred == 1).sum()
    FP = np.logical_and(Y_true == 0, Y_pred == 1).sum()

    N_images = len(results)
    miss_rate = FN / (FN + TP) if (FN + TP) > 0 else 0
    fppi = FP / N_images

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

    table = []
    for j, p in enumerate(parts):
        acc = accuracy_score(Y_true[:, j], Y_pred[:, j])
        prec = precision_score(Y_true[:, j], Y_pred[:, j], zero_division=0)
        rec = recall_score(Y_true[:, j], Y_pred[:, j], zero_division=0)
        f1s = f1_score(Y_true[:, j], Y_pred[:, j], zero_division=0)
        table.append(
            [idx_to_part[p], f"{acc:.3f}", f"{prec:.3f}", f"{rec:.3f}", f"{f1s:.3f}"]
        )

    print("[METRIC-TABLE] Per-Part Evaluation")
    print(
        tabulate(
            table, headers=["Part", "Acc", "Prec", "Rec", "F1"], tablefmt="fancy_grid"
        )
    )


class RelationProposalNetwork(nn.Module):
    def __init__(self, in_c, hidden_c=256):
        super().__init__()
        self.fc1 = nn.Linear(in_c * 2 + 4, hidden_c)
        self.fc2 = nn.Linear(hidden_c, 1)

    def forward(self, feats, boxes):
        N, C = feats.size()
        f1 = feats.unsqueeze(1).expand(-1, N, -1)
        f2 = feats.unsqueeze(0).expand(N, -1, -1)
        b1 = boxes.unsqueeze(1).expand(-1, N, -1)
        b2 = boxes.unsqueeze(0).expand(N, -1, -1)
        geom = torch.abs(b1 - b2)
        x = torch.relu(self.fc1(torch.cat([f1, f2, geom], dim=-1)))
        return self.fc2(x).squeeze(-1)


class AttentionalGCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_c, hid_c, heads=heads)
        self.gat2 = GATConv(hid_c * heads, out_c, heads=1)

    def forward(self, x, edge_index):
        x = torch.relu(self.gat1(x, edge_index))
        return self.gat2(x, edge_index)


class GraphHallucinationRCNN(nn.Module):
    def __init__(self, num_parts, topk=3, gcn_hidden=256):
        super().__init__()
        self.detector = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        in_feats = self.detector.roi_heads.box_predictor.cls_score.in_features

        self.detector.roi_heads.box_predictor = FastRCNNPredictor(
            in_feats, num_parts + 1
        )

        self.topk = topk
        self.repn = RelationProposalNetwork(in_feats)

        self.gat1 = GATConv(in_feats + 4, gcn_hidden, heads=4)
        self.gat2 = GATConv(gcn_hidden * 4, 2, heads=1)

        self.num_parts = num_parts

    def forward(self, images, targets=None):
        det_outs = self.detector(images, targets)

        if self.training:
            loss_dict = det_outs
        else:
            loss_dict = {}

        imgs, img_sizes = self.detector.transform(images)
        feats = self.detector.backbone(imgs.tensors)
        proposals, _ = self.detector.rpn(imgs, feats, targets)
        pooled = self.detector.roi_heads.box_roi_pool(
            feats, proposals, [i.shape[-2:] for i in images]
        )
        node_feats = self.detector.roi_heads.box_head(pooled)

        if self.training:
            losses_gcn = []
            for feats_img, boxes_img, img, tgt in zip(
                torch.split(node_feats, [len(p) for p in proposals], dim=0),
                proposals,
                images,
                targets,
            ):
                if boxes_img.numel() == 0:
                    continue
                geom = self._box_geom(boxes_img, img.shape[-2:])
                x = torch.cat([feats_img, geom], dim=1)
                rel = self.repn(feats_img, boxes_img)
                edge = self._make_edge_index(rel)
                h1 = nn.functional.relu(self.gat1(x, edge))
                logits = self.gat2(h1, edge)
                iou = box_iou(boxes_img, tgt["boxes"])
                best = iou.argmax(dim=1)
                miss = tgt["is_missing"][best]
                losses_gcn.append(nn.functional.cross_entropy(logits, miss))
            loss_gcn = (
                torch.stack(losses_gcn).mean()
                if losses_gcn
                else torch.tensor(0.0, device=node_feats.device)
            )
            loss_dict["loss_gcn"] = loss_gcn
            return loss_dict

        else:
            outputs = []
            idx = 0
            for img, det in zip(images, det_outs):
                Ni = len(det["boxes"])
                feats_img = node_feats[idx : idx + Ni]
                idx += Ni
                boxes_img = det["boxes"]
                geom = self._box_geom(boxes_img, img.shape[-2:])
                x = torch.cat([feats_img, geom], dim=1)
                rel = self.repn(feats_img, boxes_img)
                edge = self._make_edge_index(rel)
                h1 = nn.functional.relu(self.gat1(x, edge))
                logits = self.gat2(h1, edge)
                prob_miss = nn.functional.softmax(logits, dim=1)[:, 1]

                keep = prob_miss > 0.5
                boxes_kept = boxes_img[keep]
                parts = det["labels"][keep]
                scores = prob_miss[keep]
                final_b, final_s, final_l = [], [], []
                for c in parts.unique():
                    mask = parts == c
                    b, s = boxes_kept[mask], scores[mask]
                    sel = nms(b, s, iou_threshold=0.5)
                    final_b.append(b[sel])
                    final_s.append(s[sel])
                    final_l.append(torch.full_like(s[sel], c))
                outputs.append(
                    {
                        "boxes_missing": (
                            torch.cat(final_b)
                            if final_b
                            else torch.zeros((0, 4), device=img.device)
                        ),
                        "scores_missing": (
                            torch.cat(final_s)
                            if final_s
                            else torch.zeros((0,), device=img.device)
                        ),
                        "labels_missing": (
                            torch.cat(final_l)
                            if final_l
                            else torch.zeros((0,), device=img.device)
                        ),
                    }
                )
            return outputs

    def _make_edge_index(self, rel):
        N = rel.size(0)
        rn = rel - torch.eye(N, device=rel.device) * 1e6
        idx = rn.topk(self.topk, dim=1).indices.flatten()
        dst = (
            torch.arange(N, device=rel.device)
            .unsqueeze(1)
            .expand(-1, self.topk)
            .flatten()
        )
        return torch.stack([dst, idx], dim=0)

    def _box_geom(self, boxes, shape):
        h, w = shape
        nb = boxes.clone()
        nb[:, [0, 2]] /= w
        nb[:, [1, 3]] /= h
        return torch.stack(
            [nb[:, 0], nb[:, 1], nb[:, 2] - nb[:, 0], nb[:, 3] - nb[:, 1]], dim=1
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphHallucinationRCNN(num_parts=len(train_dataset.all_parts), topk=3).to(
    device
)
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
                for t in targets:
                    t["boxes"]       = t["boxes"].to(device)
                    t["labels"]      = t["labels"].to(device)
                    t["is_missing"]  = t["is_missing"].to(device)

                start_time = time.time()

                optimizer.zero_grad()
                loss_dict = model(images, targets)
                total_loss = sum(loss_dict.values())
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
            results = evaluate_model(
                model, valid_loader, train_dataset.all_parts, device
            )
            parts = train_dataset.all_parts
            Y_true = np.array(
                [
                    [1 if p in r["true_missing_parts"] else 0 for p in parts]
                    for r in results
                ]
            )
            Y_pred = np.array(
                [
                    [1 if p in r["predicted_missing_parts"] else 0 for p in parts]
                    for r in results
                ]
            )
            macro_f1 = f1_score(Y_true, Y_pred, average="macro", zero_division=0)

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                no_improve = 0
                torch.save(
                    model.state_dict(),
                    "/var/scratch/sismail/models/graph_rcnn/graphrcnn_MobileNet_missing_baseline_model.pth",
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

if torch.cuda.is_available():
    nvmlShutdown()


model.load_state_dict(
    torch.load(
        "/var/scratch/sismail/models/graph_rcnn/graphrcnn_MobileNet_missing_baseline_model.pth",
        map_location=device,
    )
)
model.to(device)

model.eval()

results_per_image = evaluate_model(model, valid_loader, valid_dataset.all_parts, device)


part_level_evaluation(
    results_per_image, train_dataset.part_to_idx, train_dataset.idx_to_part
)

visualize_and_save_predictions(
    model,
    valid_dataset,
    device,
    out_dir="/home/sismail/Thesis/visualisations/",
    n_images=5,
)


results_per_image = evaluate_model(model, test_loader, test_dataset.all_parts, device)

part_level_evaluation(
    results_per_image, train_dataset.part_to_idx, train_dataset.idx_to_part
)
