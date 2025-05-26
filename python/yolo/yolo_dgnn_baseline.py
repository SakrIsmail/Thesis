import os
import shutil
import json
import random
import time
import psutil
import gc
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import logging
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import roi_align
from ultralytics import YOLO
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlShutdown,
)
from codecarbon import EmissionsTracker
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("ultralytics.yolo").setLevel(logging.ERROR)


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
        self.part_to_idx = {part: idx for idx, part in enumerate(self.all_parts)}
        self.idx_to_part = {idx: part for idx, part in enumerate(self.all_parts)}
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_times, gpu_mem, cpu_mem = [], [], []
nvml_handle, em_tracker = None, None
best_macro_f1, no_imp = 0.0, 0


def on_epoch_start(trainer):
    global batch_times, gpu_mem, cpu_mem, nvml_handle, em_tracker
    batch_times.clear()
    gpu_mem.clear()
    cpu_mem.clear()
    em_tracker = EmissionsTracker(log_level="critical", save_to_file=False).__enter__()
    if trainer.device.type == "cuda":
        nvmlInit()
        nvml_handle = nvmlDeviceGetHandleByIndex(0)


def on_batch_start(trainer):
    trainer._batch_start = time.time()


def on_batch_end(trainer):
    global batch_times, gpu_mem, cpu_mem
    t = time.time() - trainer._batch_start
    batch_times.append(t)
    if nvml_handle:
        mi = nvmlDeviceGetMemoryInfo(nvml_handle)
        gpu_mem.append(mi.used // (1024**2))
    else:
        gpu_mem.append(0)
    cpu_mem.append(psutil.virtual_memory().used // (1024**2))
    gc.collect()


def on_epoch_end(trainer):
    global best_macro_f1, no_imp
    energy = em_tracker.__exit__(None, None, None).final_emissions_data.energy_consumed
    co2 = em_tracker.final_emissions
    nvmlShutdown()
    # print table
    stats = [
        ["Epoch", trainer.epoch],
        ["Loss", f"{trainer.loss:.4f}"],
        ["Avg batch time", f"{np.mean(batch_times):.3f}s"],
        ["Max GPU", f"{max(gpu_mem)}MB"],
        ["Max CPU", f"{max(cpu_mem)}MB"],
        ["Energy", f"{energy:.4f}kWh"],
        ["CO₂", f"{co2:.4f}kg"],
    ]
    print(tabulate(stats, headers=["Metric", "Value"], tablefmt="pretty"))
    # validation missing-part F1
    model_yaml = os.path.join(
        trainer.args.project, trainer.args.name, "weights", "last.pt"
    )
    val_model = YOLO(model_yaml).to(trainer.device).eval()
    res = run_yolo_inference(
        val_model,
        valid_loader,
        valid_dataset.part2idx,
        valid_dataset.idx2part,
        trainer.device,
    )
    parts = list(valid_dataset.part2idx.values())
    Yt = np.array([[int(p in r["true_missing_parts"]) for p in parts] for r in res])
    Yp = np.array(
        [[int(p in r["predicted_missing_parts"]) for p in parts] for r in res]
    )
    m_f1 = f1_score(Yt, Yp, average="macro", zero_division=0)
    print(f"Val Macro-F1: {m_f1:.4f}")
    # early stop
    if m_f1 > best_macro_f1:
        best_macro_f1 = m_f1
        no_imp = 0
        shutil.copy(model_yaml, model_yaml.replace("last.pt", "best.pt"))
    else:
        no_imp += 1
        if no_imp >= 5:
            trainer.stop_training = True


class YOLOGNNWrapper(nn.Module):
    def __init__(
        self,
        yolo_path,
        hidden_dim=128,
        num_parts=22,
        dist_thresh=50.0,
        cos_thresh=0.8,
        roi_size=7,
    ):
        super().__init__()
        yolo = YOLO(yolo_path)
        m = list(yolo.model.model)
        self.extractor = nn.Sequential(*m[:-1])
        self.head = m[-1]
        # GNN placeholders
        self.gcn1 = self.gcn2 = self.classifier = None
        self.hidden = hidden_dim
        self.num_parts = num_parts
        self.dth, self.cth = dist_thresh, cos_thresh
        self.roi_s = (roi_size, roi_size)

    def _init_gnn(self, in_dim):
        self.gcn1 = GCNConv(in_dim, self.hidden)
        self.gcn2 = GCNConv(self.hidden, self.hidden)
        self.classifier = nn.Linear(self.hidden, self.num_parts)

    def forward(self, imgs):
        B = imgs.size(0)
        fmap = self.extractor(imgs)  # [B,C,H,W]
        dets = self.head(fmap)  # list len B
        node_graphs = []
        for b, det in enumerate(dets):
            xy = dets[b].boxes.xyxyn.cpu() * 640
            if xy.numel() == 0:
                continue
            rois = torch.cat(
                [torch.full((xy.size(0), 1), b), xy.to(imgs.device)], dim=1
            )
            p = roi_align(fmap, rois, self.roi_s)  # [N,C,7,7]
            N, C, h, w = p.shape
            fv = p.view(N, -1)
            fv_norm = F.normalize(fv, dim=-1)
            ctr = (xy[:, :2] + xy[:, 2:]) / 2
            if self.gcn1 is None:
                self._init_gnn(fv.size(1) + 2)
            x = torch.cat([fv.to(imgs.device), ctr.to(imgs.device)], dim=1)
            d = torch.cdist(ctr, ctr)
            c = fv_norm @ fv_norm.T
            edges = ((d < self.dth) | (c > self.cth)).nonzero().t().contiguous()
            batch_idx = torch.full((N,), b, dtype=torch.long, device=x.device)
            node_graphs.append(Data(x=x, edge_index=edges, batch=batch_idx))
        if not node_graphs:
            return dets, torch.zeros(B, self.num_parts, device=imgs.device)
        batch = Batch.from_data_list(node_graphs)
        h = F.relu(self.gcn1(batch.x, batch.edge_index))
        h = F.relu(self.gcn2(h, batch.edge_index))
        hg = global_mean_pool(h, batch.batch)
        logits = self.classifier(hg)
        # pad back to B
        if logits.size(0) < B:
            out = torch.zeros(B, self.num_parts, device=logits.device)
            out[: logits.size(0)] = logits
            logits = out
        return dets, logits


def run_yolo_inference(model, loader, p2i, i2p, device):
    model.model.to(device).eval()
    results = []
    for imgs, tgs in tqdm(loader, desc="Eval"):
        arrs = [
            (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8) for img in imgs
        ]
        preds = model(arrs, device=device, verbose=False)
        # same logic as before...
        for i, det in enumerate(preds):
            pr = set(det.boxes.cls.cpu().numpy().astype(int).tolist())
            tm = set(tgs[i]["missing_labels"].tolist())
            ap = set(p2i.values())
            results.append(
                {
                    "image_id": tgs[i]["image_id"].item(),
                    "predicted_missing_parts": ap - pr,
                    "true_missing_parts": tm,
                }
            )
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


model = YOLOGNNWrapper(
    "yolov8m.pt",
    hidden_dim=128,
    num_parts=len(train_dataset.all_parts),
    dist_thresh=50.0,
    cos_thresh=0.8,
    roi_size=7,
).to(device)

model.head.add_callback("on_train_epoch_start", on_epoch_start)
model.head.add_callback("on_train_batch_start", on_batch_start)
model.head.add_callback("on_train_batch_end", on_batch_end)
model.head.add_callback("on_train_epoch_end", on_epoch_end)

model.train(
    data="/var/scratch/sismail/data/yolo_format/noaug/data.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    optimizer="AdamW",
    lr0=1e-4,
    weight_decay=1e-4,
    workers=4,
    device=device,
    seed=42,
    verbose=False,
    plots=False,
    project="/var/scratch/sismail/models/yolo/runs",
    name="bikeparts_experiment_gnn_baseline",
    exist_ok=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLOGNNWrapper(
    yolo_path="yolov8m.pt",
    hidden_dim=128,
    num_parts=len(train_dataset.all_parts),
    dist_thresh=50.0,
    cos_thresh=0.8,
    roi_size=7,
).to(device)


best_ckpt = (
    "/var/scratch/$USER/models/yolo/runs/bikeparts_experiment_gnn/weights/best.pt"
)
model.head.model.load_state_dict(
    torch.load(best_ckpt, map_location=device)
)  # detection head
model.load_state_dict(
    torch.load(best_ckpt.replace("best.pt", "best_wrapper.pt"), map_location=device)
)
model.eval()


model.eval()

val_results = run_yolo_inference(
    model, valid_loader, valid_dataset.part_to_idx, valid_dataset.idx_to_part, device
)
test_results = run_yolo_inference(
    model, test_loader, test_dataset.part_to_idx, test_dataset.idx_to_part, device
)

part_level_evaluation(val_results, valid_dataset.part_to_idx, valid_dataset.idx_to_part)
part_level_evaluation(test_results, test_dataset.part_to_idx, test_dataset.idx_to_part)
