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
from tabulate import tabulate
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ultralytics import YOLO
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlShutdown,
)
from codecarbon import EmissionsTracker

logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("ultralytics.yolo").setLevel(logging.ERROR)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    Args:
        seed (int): Seed value to set for random number generators.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


def seed_worker(worker_id):
    """
    Set the seed for each worker to ensure reproducibility.
    Args:
        worker_id (int): The ID of the worker.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


final_output_json = (
    "data/processed/final_annotations_without_occluded.json"
)
image_directory = "data/images"

# Split the dataset into train, validation, and test sets
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
    """
    Custom dataset for bike parts detection.
    Args:
        annotations_dict (dict): Dictionary containing annotations with keys 'all_parts' and 'images'.
        image_dir (str): Directory containing the images.
        transform (callable, optional): A function/transform to apply to the images.
        augment (bool): Whether to apply data augmentation.
        target_size (tuple): The target size for resizing images.
    """

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
        """
        Apply random augmentations to the image and bounding boxes.
        Args:
            image (PIL.Image): The input image.
            boxes (torch.Tensor): Bounding boxes in the format [x_min, y_min, x_max, y_max].
        Returns:
            PIL.Image: Augmented image.
            torch.Tensor: Augmented bounding boxes.
        """
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
        """
        Get an item from the dataset.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            tuple: A tuple containing the image and its corresponding target dictionary.
        """
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
batch_times, gpu_memories, cpu_memories = [], [], []
batch_count = 0
nvml_handle, em_tracker = None, None
yolo_scheduler = None
start_time = 0
best_macro_f1 = 0.0
no_improve_epochs = 0
patience = 8


def on_train_start(trainer):
    """
    Callback function to initialize the YOLO scheduler and other variables at the start of training.
    Args:
        trainer (YOLO): The YOLO trainer instance.
    """
    global yolo_scheduler
    optim = trainer.optimizer

    if yolo_scheduler is None:
        yolo_scheduler = ReduceLROnPlateau(
            optim,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=False,
        )


def on_train_epoch_start(trainer):
    """
    Callback function to initialize variables at the start of each training epoch.
    Args:
        trainer (YOLO): The YOLO trainer instance.
    """
    global batch_times, gpu_memories, cpu_memories, nvml_handle, em_tracker, batch_count

    batch_times.clear()
    gpu_memories.clear()
    cpu_memories.clear()
    batch_count = 0

    em_tracker = EmissionsTracker(log_level="critical", save_to_file=False)
    em_tracker.__enter__()

    if trainer.device.type == "cuda":
        nvmlInit()
        nvml_handle = nvmlDeviceGetHandleByIndex(0)


def on_train_batch_start(trainer):
    """
    Callback function to initialize variables at the start of each training batch.
    Args:
        trainer (YOLO): The YOLO trainer instance.
    """
    global start_time

    start_time = time.time()


def on_train_batch_end(trainer):
    """
    Callback function to log metrics at the end of each training batch.
    Args:
        trainer (YOLO): The YOLO trainer instance.
    """
    global batch_count, start_time

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

    print(
        f"Batch {batch_count} | loss={trainer.loss:.4f} | time={inference_time:.3f}s | "
        f"GPU={gpu_memories[-1]:.0f}MB | CPU={cpu_memories[-1]:.0f}MB",
        file=sys.stderr,
    )
    gc.collect()


def on_train_epoch_end(trainer):
    """
    Callback function to log metrics at the end of each training epoch.
    Args:
        trainer (YOLO): The YOLO trainer instance.
    """
    global nvml_handle, em_tracker

    em_tracker.__exit__(None, None, None)
    energy = em_tracker.final_emissions_data.energy_consumed
    co2 = em_tracker.final_emissions

    if nvml_handle:
        nvmlShutdown()
    table = [
        ["Epoch", trainer.epoch],
        ["Final Loss", f"{trainer.loss:.4f}"],
        ["Avg Batch Time (s)", f"{np.mean(batch_times):.4f}"],
        ["Max GPU Mem (MB)", f"{np.max(gpu_memories):.1f}"],
        ["Max CPU Mem (MB)", f"{np.max(cpu_memories):.1f}"],
        ["Energy (kWh)", f"{energy:.4f}"],
        ["CO₂ (kg)", f"{co2:.4f}"],
    ]
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="pretty"))


def on_model_save(trainer):
    """
    Callback function to evaluate the model and save the best weights based on macro F1 score.
    Args:
        trainer (YOLO): The YOLO trainer instance.
    """
    global best_macro_f1, no_improve_epochs, yolo_scheduler, patience, valid_loader, device

    wdir = os.path.join(trainer.args.project, trainer.args.name, "weights")
    last_path = os.path.join(wdir, "last.pt")
    model = YOLO(last_path)
    model.to(device).eval()
    results = run_yolo_inference(
        model,
        valid_loader,
        valid_dataset.part_to_idx,
        valid_dataset.idx_to_part,
        device,
    )

    parts = list(valid_dataset.part_to_idx.values())
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

    yolo_scheduler.step(macro_f1)

    print(f"Epoch {trainer.epoch + 1}: Macro F1 Score = {macro_f1:.4f}")

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        no_improve_epochs = 0
        shutil.copy(last_path, os.path.join(wdir, "best.pt"))
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {trainer.epoch + 1}")
            trainer.stop = True


def run_yolo_inference(model, loader, part_to_idx, idx_to_part, device):
    """
    Run inference on the YOLO model and collect results.
    Args:
        model (YOLO): The YOLO model instance.
        loader (DataLoader): DataLoader for the dataset.
        part_to_idx (dict): Mapping from part names to indices.
        idx_to_part (dict): Mapping from indices to part names.
        device (torch.device): Device to run the model on.
    Returns:
        list: List of dictionaries containing inference results.
    """
    model.model.to(device).eval()
    results = []

    for images, targets in tqdm(loader, desc="Eval"):
        np_images = []
        for img in images:
            arr = img.cpu().permute(1, 2, 0).numpy()
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            np_images.append(arr)

        preds = model(np_images, device=device, verbose=False)

        for i, det in enumerate(preds):
            pred_labels = set(det.boxes.cls.cpu().numpy().astype(int).tolist())
            true_missing = set(targets[i]["missing_labels"].tolist())
            all_parts = set(part_to_idx.values())
            results.append(
                {
                    "image_id": targets[i]["image_id"].item(),
                    "predicted_missing_parts": all_parts - pred_labels,
                    "true_missing_parts": true_missing,
                }
            )
    return results


def part_level_evaluation(results, part_to_idx, idx_to_part):
    """
    Evaluate the model's performance on a per-part basis.
    Args:
        results (list): List of dictionaries containing evaluation results for each image.
        part_to_idx (dict): Mapping from part names to indices.
        idx_to_part (dict): Mapping from indices to part names.
    """
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


model = YOLO("yolov8m.pt", verbose=False)
model.add_callback("on_train_start", on_train_start)
model.add_callback("on_train_epoch_start", on_train_epoch_start)
model.add_callback("on_train_batch_start", on_train_batch_start)
model.add_callback("on_train_batch_end", on_train_batch_end)
model.add_callback("on_train_epoch_end", on_train_epoch_end)
model.add_callback("on_model_save", on_model_save)
model.to(device)

model.train(
    data="data/yolo_format/noaug/data.yaml",
    epochs=100,
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
    project="models/yolo/runs",
    name="bikeparts_experiment_baseline",
    exist_ok=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(
    "models/yolo/runs/bikeparts_experiment_baseline/weights/best.pt"
)
model.to(device)


model.eval()

val_results = run_yolo_inference(
    model, valid_loader, valid_dataset.part_to_idx, valid_dataset.idx_to_part, device
)
test_results = run_yolo_inference(
    model, test_loader, test_dataset.part_to_idx, test_dataset.idx_to_part, device
)

part_level_evaluation(val_results, valid_dataset.part_to_idx, valid_dataset.idx_to_part)
part_level_evaluation(test_results, test_dataset.part_to_idx, test_dataset.idx_to_part)
