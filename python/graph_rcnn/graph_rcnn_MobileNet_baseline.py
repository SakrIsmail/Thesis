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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


final_output_json = (
    "/var/scratch/sismail/data/processed/final_annotations_without_occluded.json"
)
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


def evaluate_model(model, data_loader, part_to_idx, device):
    model.eval()

    all_parts_set = set(part_to_idx.values())
    results_per_image = []

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)

        for i in range(len(images)):
            pred_parts = set(predictions[i]["labels"].cpu().numpy().tolist())
            true_missing_parts = set(
                targets[i]["missing_labels"].cpu().numpy().tolist()
            )
            image_id = targets[i]["image_id"].item()

            predicted_missing_parts = all_parts_set - pred_parts

            results_per_image.append(
                {
                    "image_id": image_id,
                    "predicted_missing_parts": predicted_missing_parts,
                    "true_missing_parts": true_missing_parts,
                }
            )

    return results_per_image


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


detector = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
in_feats = detector.roi_heads.box_predictor.cls_score.in_features

detector.roi_heads.box_predictor = FastRCNNPredictor(
    in_feats, len(train_dataset.all_parts) + 1
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


class GraphRCNN(nn.Module):
    def __init__(self, detector, num_classes, k=5):
        super().__init__()
        self.detector = detector
        in_feats = detector.roi_heads.box_predictor.cls_score.in_features
        self.repn = RelationProposalNetwork(in_feats)
        self.agcn = AttentionalGCN(in_feats + 4, 256, num_classes)
        self.k = k
        self.log_sigma_gcn = nn.Parameter(torch.zeros(()))
        self.log_sigma_repnet = nn.Parameter(torch.zeros(()))

    def _get_roi_feats(self, img, boxes):
        fmap = self.detector.backbone(img)
        roi = self.detector.roi_heads.box_roi_pool(fmap, [boxes], [img.shape[-2:]])
        return self.detector.roi_heads.box_head(roi)

    def _make_edge_index(self, scores):
        N = scores.size(0)
        device = scores.device

        if N == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        if N == 1:
            return torch.tensor([[0], [0]], dtype=torch.long, device=device)

        if N <= self.k:
            row = torch.arange(N, device=device).unsqueeze(1).expand(N, N).flatten()
            col = torch.arange(N, device=device).unsqueeze(0).expand(N, N).flatten()
            return torch.stack([row, col], dim=0).long()

        topk = torch.topk(scores, self.k + 1, dim=1)
        neighbors = topk.indices[:, 1:]
        src = torch.arange(N, device=device).unsqueeze(1).expand(-1, self.k).flatten()
        dst = neighbors.flatten()
        return torch.stack([src, dst], dim=0).long()

    def _box_geom(self, boxes, shape):
        h, w = shape
        nb = boxes.clone()
        nb[:, [0, 2]] /= w
        nb[:, [1, 3]] /= h
        return torch.stack(
            [nb[:, 0], nb[:, 1], nb[:, 2] - nb[:, 0], nb[:, 3] - nb[:, 1]], dim=1
        )

    def forward(self, images, targets=None):
        if self.training:
            loss_dict = self.detector(images, targets)
            base_loss = sum(loss_dict.values())

            gcn_loss, repnet_loss = self.compute_gcn_loss(images, targets)

            total_loss = base_loss + gcn_loss + repnet_loss
            loss_dict.update(
                {
                    "base_loss": base_loss,
                    "gcn_loss": gcn_loss,
                    "repnet_loss": repnet_loss,
                }
            )
            return total_loss, loss_dict

        dets = self.detector(images)
        outputs = []
        for img, det in zip(images, dets):
            boxes = det.get("boxes", torch.empty((0, 4), device=img.device))
            scores = det.get("scores", torch.empty((0,), device=img.device))
            labels = det.get(
                "labels", torch.empty((0,), dtype=torch.long, device=img.device)
            )

            Ni = boxes.shape[0]
            if Ni == 0:

                outputs.append(
                    {
                        "boxes": torch.empty((0, 4), device=img.device),
                        "scores": torch.empty((0,), device=img.device),
                        "labels": torch.empty(
                            (0,), dtype=torch.long, device=img.device
                        ),
                    }
                )
                continue

            feats = self._get_roi_feats(img.unsqueeze(0), boxes).squeeze(0)
            rel_scores = self.repn(feats, boxes)
            edge_index = self._make_edge_index(rel_scores)

            geom = self._box_geom(boxes, img.shape[-2:])
            node_feats = torch.cat([feats, geom], dim=1)
            logits = self.agcn(node_feats, edge_index)
            probs = torch.softmax(logits, dim=1)

            new_scores, new_labels = probs.max(dim=1)
            keep = (new_labels != 0) & (new_scores > 0.5)

            if keep.sum() == 0:
                outputs.append(
                    {
                        "boxes": torch.empty((0, 4), device=img.device),
                        "scores": torch.empty((0,), device=img.device),
                        "labels": torch.empty(
                            (0,), dtype=torch.long, device=img.device
                        ),
                    }
                )
                continue

            boxes2 = boxes[keep]
            scores2 = new_scores[keep]
            labels2 = new_labels[keep]

            keep_idx = nms(boxes2, scores2, iou_threshold=0.5)
            boxes_n = boxes2[keep_idx]
            scores_n = scores2[keep_idx]
            labels_n = labels2[keep_idx]

            final_boxes, final_scores, final_labels = [], [], []
            for c in labels_n.unique():
                mask = labels_n == c
                c_scores = scores_n[mask]
                c_boxes = boxes_n[mask]
                best = c_scores.argmax()
                final_boxes.append(c_boxes[best])
                final_scores.append(c_scores[best])
                final_labels.append(c)
            outputs.append(
                {
                    "boxes": torch.stack(final_boxes),
                    "scores": torch.stack(final_scores),
                    "labels": torch.stack(final_labels),
                }
            )
        return outputs

    def compute_gcn_loss(self, images, targets):
        gcn_logits_list = []
        gcn_labels_list = []
        rep_sum = 0.0
        rep_count = 0

        for img, tgt in zip(images, targets):
            boxes = tgt["boxes"]
            labels = tgt["labels"]

            Ni = boxes.shape[0]
            if Ni == 0:
                continue

            feats = self._get_roi_feats(img.unsqueeze(0), boxes)

            rel = self.repn(feats, boxes)

            edge_index = self._make_edge_index(rel)

            geom = self._box_geom(boxes, img.shape[-2:])
            node_feats = torch.cat([feats, geom], dim=1)

            logits = self.agcn(node_feats, edge_index)
            gcn_logits_list.append(logits)
            gcn_labels_list.append(labels)

            iou_mat = box_iou(boxes, boxes)
            target_matrix = (iou_mat > 0.1).float()

            this_rep_loss = nn.functional.binary_cross_entropy_with_logits(
                rel, target_matrix, reduction="mean"
            )
            rep_sum += this_rep_loss
            rep_count += 1

        if rep_count == 0:
            zero = torch.tensor(0.0, device=images[0].device)
            return zero, zero
        all_logits = torch.cat(gcn_logits_list, dim=0)
        all_labels = torch.cat(gcn_labels_list, dim=0)

        gcn_loss = nn.functional.cross_entropy(all_logits, all_labels)

        repnet_loss = rep_sum / rep_count

        return gcn_loss, repnet_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = GraphRCNN(detector, len(train_dataset.all_parts) + 1).to(device)

detector_params = (
    list(model.detector.backbone.parameters())
    + list(model.detector.rpn.parameters())
    + list(model.detector.roi_heads.parameters())
)
graph_params = list(model.repn.parameters()) + list(model.agcn.parameters())

opt_det = torch.optim.AdamW(detector_params, lr=1e-4, weight_decay=1e-4)
sched_det = ReduceLROnPlateau(
    opt_det, mode="max", factor=0.5, patience=3, min_lr=1e-6, verbose=True
)

opt_graph = torch.optim.AdamW(graph_params, lr=1e-4, weight_decay=1e-4)

sched_graph = ReduceLROnPlateau(
    opt_graph, mode="max", factor=0.5, patience=3, min_lr=1e-6, verbose=True
)

epochs = 2
freeze_epoch = 2
patience = 8
detector_best_macro_f1 = 0
detector_no_improve = 0
joint_best_macro_f1 = 0
joiny_no_improve = 0

if torch.cuda.is_available():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

for p in graph_params:
    p.requires_grad = False

for epoch in range(1, epochs + 1):
    if epoch < freeze_epoch:
        with EmissionsTracker(log_level="critical", save_to_file=False) as tracker:
            model.detector.train()
            batch_times, gpu_memories, cpu_memories = [], [], []
            with tqdm(
                train_loader,
                unit="batch",
                desc=f"Detector Epoch {epoch}/{freeze_epoch}",
            ) as tepoch:
                for images, targets in tepoch:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    start_time = time.time()

                    opt_det.zero_grad()
                    loss_dict = model.detector(images, targets)
                    total_loss = sum(loss_dict.values())
                    total_loss.backward()
                    opt_det.step()

                    end_time = time.time()
                    inference_time = end_time - start_time

                    batch_times.append(inference_time)
                    if torch.cuda.is_available():
                        gpu_mem_used = nvmlDeviceGetMemoryInfo(handle).used / 1024**2
                        gpu_memories.append(gpu_mem_used)
                    else:
                        gpu_mem_used = 0

                    cpu_mem_used = psutil.virtual_memory().used / 1024**2
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

        avg_time = np.mean(batch_times)
        max_gpu = max(gpu_memories) if gpu_memories else 0
        max_cpu = max(cpu_memories)
        energy = tracker.final_emissions_data.energy_consumed
        co2 = tracker.final_emissions
        print(
            tabulate(
                [
                    ["Detector Epoch", epoch],
                    ["Detector Loss", f"{total_loss.item():.4f}"],
                    ["Avg Batch Time (s)", f"{avg_time:.3f}"],
                    ["Max GPU Mem (MB)", f"{max_gpu:.0f}"],
                    ["Max CPU Mem (MB)", f"{max_cpu:.0f}"],
                    ["Energy (kWh)", f"{energy:.4f}"],
                    ["CO2 (kg)", f"{co2:.4f}"],
                ],
                headers=["Metric", "Value"],
                tablefmt="pretty",
            )
        )

        model.eval()
        results = evaluate_model(
            model.detector, valid_loader, train_dataset.part_to_idx, device
        )
        parts = list(train_dataset.part_to_idx.values())
        Y_true = np.array(
            [[1 if p in r["true_missing_parts"] else 0 for p in parts] for r in results]
        )
        Y_pred = np.array(
            [
                [1 if p in r["predicted_missing_parts"] else 0 for p in parts]
                for r in results
            ]
        )
        macro_f1 = f1_score(Y_true, Y_pred, average="macro", zero_division=0)

        sched_det.step(macro_f1)

        if macro_f1 > detector_best_macro_f1:
            detector_best_macro_f1 = macro_f1
            detector_no_improve = 0
            torch.save(
                model.detector.state_dict(),
                "/var/scratch/sismail/models/graph_rcnn/graphrcnn_detector_baseline_model.pth",
            )
        else:
            detector_no_improve += 1
            if detector_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                freeze_epoch = epoch

        continue

    if epoch == freeze_epoch:
        model.detector.load_state_dict(
            torch.load(
                "/var/scratch/sismail/models/graph_rcnn/graphrcnn_detector_baseline_model.pth"
            )
        )

        for p in detector_params:
            p.requires_grad = False
        for p in graph_params:
            p.requires_grad = True

    with EmissionsTracker(log_level="critical", save_to_file=False) as tracker:
        model.train()
        batch_times, gpu_memories, cpu_memories = [], [], []

        with tqdm(
            train_loader,
            unit="batch",
            desc=f"Joint Epoch {epoch - freeze_epoch + 1}/{epochs - freeze_epoch + 1}",
        ) as tepoch:
            for images, targets in tepoch:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                start_time = time.time()
                opt_graph.zero_grad()
                total_loss, loss_dict = model(images, targets)
                total_loss.backward()
                opt_graph.step()
                end_time = time.time()

                inference_time = end_time - start_time
                batch_times.append(inference_time)
                if torch.cuda.is_available():
                    gpu_mem_used = nvmlDeviceGetMemoryInfo(handle).used / 1024**2
                    gpu_memories.append(gpu_mem_used)
                else:
                    gpu_mem_used = 0

                cpu_mem_used = psutil.virtual_memory().used / 1024**2
                cpu_memories.append(cpu_mem_used)

                tepoch.set_postfix(
                    {
                        "loss": f"{total_loss.item():.4f}",
                        "time (s)": f"{inference_time:.3f}",
                        "GPU Mem (MB)": f"{gpu_mem_used:.0f}",
                        "CPU Mem (MB)": f"{cpu_mem_used:.0f}",
                    }
                )

    avg_time = np.mean(batch_times)
    max_gpu = max(gpu_memories) if gpu_memories else 0
    max_cpu = max(cpu_memories)
    energy = tracker.final_emissions_data.energy_consumed
    co2 = tracker.final_emissions
    print(
        tabulate(
            [
                ["Joint Epoch", epoch - freeze_epoch + 1],
                ["GNN Loss", f"{total_loss.item():.4f}"],
                ["Avg Batch Time (s)", f"{avg_time:.3f}"],
                ["Max GPU Mem (MB)", f"{max_gpu:.0f}"],
                ["Max CPU Mem (MB)", f"{max_cpu:.0f}"],
                ["Energy (kWh)", f"{energy:.4f}"],
                ["CO2 (kg)", f"{co2:.4f}"],
            ],
            headers=["Metric", "Value"],
            tablefmt="pretty",
        )
    )

    model.eval()
    print(f"\nEvaluating on validation set after Epoch {epoch + 1}...")
    results_per_image = evaluate_model(
        model, valid_loader, train_dataset.part_to_idx, device
    )

    parts = list(train_dataset.part_to_idx.values())
    Y_true = np.array(
        [
            [1 if p in r["true_missing_parts"] else 0 for p in parts]
            for r in results_per_image
        ]
    )
    Y_pred = np.array(
        [
            [1 if p in r["predicted_missing_parts"] else 0 for p in parts]
            for r in results_per_image
        ]
    )
    macro_f1 = f1_score(Y_true, Y_pred, average="macro", zero_division=0)

    sched_graph.step(macro_f1)

    if macro_f1 > joint_best_macro_f1:
        joint_best_macro_f1 = macro_f1
        joiny_no_improve = 0
        torch.save(
            model.state_dict(),
            f"/var/scratch/sismail/models/graph_rcnn/graphrcnn_MobileNet_baseline_model.pth",
        )
    else:
        joiny_no_improve += 1
        if joiny_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break


if torch.cuda.is_available():
    nvmlShutdown()


model.load_state_dict(
    torch.load(
        "/var/scratch/sismail/models/graph_rcnn/graphrcnn_MobileNet_baseline_model.pth",
        map_location=device,
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
