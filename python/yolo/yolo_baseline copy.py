import time
import json
import pathlib
import torch
import numpy as np
import psutil
from tqdm import tqdm
from tabulate import tabulate
from codecarbon import EmissionsTracker

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    nvml_available = True
except ImportError:
    nvml_available = False

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from ultralytics import YOLO
import torchvision.transforms as T
from PIL import Image

# -----------------------------
# Custom Dataset
# -----------------------------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, annotations, transforms=None):
        # Limit to 10 samples for testing
        self.image_paths = image_paths  # Limit to 10 images for testing
        self.annotations = annotations
        self.transforms = transforms or T.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
        ann = self.annotations.get(img_path.name, {})
        boxes = torch.tensor(ann.get('boxes', []), dtype=torch.float32)
        labels = torch.tensor(ann.get('labels', []), dtype=torch.int64)
        missing = ann.get('missing_parts', [])
        target = {'boxes': boxes, 'labels': labels, 'missing_parts': missing}
        return img, target

# -----------------------------
# Callbacks for Metrics
# -----------------------------
def on_train_epoch_begin(trainer):
    """Initialize TQDM progress bar at the start of each epoch."""
    trainer.pbar = tqdm(total=trainer.dataloader_len,
                         desc=f"Epoch {trainer.epoch+1}/{trainer.epochs}",
                         unit='batch')

def on_train_batch_end(trainer):
    """Update TQDM progress bar after each batch."""
    loss = trainer.metrics.get('loss', 0.0)
    dt = trainer.metrics.get('time', 0.0)
    trainer.pbar.set_postfix({'loss': f"{loss:.4f}", 'time': f"{dt:.3f}"})
    trainer.pbar.update(1)

def on_train_epoch_end(trainer):
    """Close the TQDM progress bar after each epoch."""
    trainer.pbar.close()

def on_train_epoch_begin_perf(trainer):
    """Start performance tracking at the beginning of each epoch."""
    trainer.epoch_gpu, trainer.epoch_cpu = [], []
    trainer.tracker = EmissionsTracker(log_level='critical', save_to_file=False)
    trainer.tracker.start()

def on_train_batch_end_perf(trainer):
    """Track memory usage after each batch."""
    if nvml_available:
        mem = nvmlDeviceGetMemoryInfo(trainer.nv).used / 1024**2  # in MB
        trainer.epoch_gpu.append(mem)
    trainer.epoch_cpu.append(psutil.virtual_memory().used / 1024**2)  # in MB

def on_train_epoch_end_perf(trainer):
    """Log performance statistics at the end of each epoch."""
    trainer.tracker.stop()
    energy, co2 = trainer.tracker.final_emissions_data.energy_consumed, trainer.tracker.final_emissions
    times = [m['time'] for m in trainer.metrics_history if 'time' in m]
    losses = [m['loss'] for m in trainer.metrics_history if 'loss' in m]
    avg_loss = np.mean(losses) if losses else 0.0
    table = [
        ['Epoch', trainer.epoch+1],
        ['Avg Loss', f"{avg_loss:.4f}"],
        ['Avg Time(s)', f"{np.mean(times):.4f}" if times else '0.0000'],
        ['Max GPU(MB)', f"{max(trainer.epoch_gpu):.0f}" if trainer.epoch_gpu else '0'],
        ['Max CPU(MB)', f"{max(trainer.epoch_cpu):.0f}" if trainer.epoch_cpu else '0'],
        ['Energy(kWh)', f"{energy:.4f}"],
        ['CO2(kg)',    f"{co2:.4f}"]
    ]
    print('\n' + tabulate(table, headers=['Metric','Value'], tablefmt='pretty'))

# -----------------------------
# Main Script
# -----------------------------
if __name__ == '__main__':
    # Load data
    ann = json.load(open('data/processed/final_annotations_without_occluded.json'))
    splits = {}    
    for split in ['train','val','test']:
        paths = sorted(pathlib.Path(f"data/yolo_format/noaug/images/{split}").glob('*.jpg'))
        ds = CustomDataset(paths, ann['images'])
        splits[split] = ds

    # Check if datasets have loaded correctly (debugging step)
    print(f"Training set size: {len(splits['train'])}")
    print(f"Validation set size: {len(splits['val'])}")
    print(f"Test set size: {len(splits['test'])}")

    # Create the YOLO model instance
    model = YOLO('yolov8n.pt')

    # Add the callbacks to the model
    model.add_callback("on_train_epoch_begin", on_train_epoch_begin)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # Add the performance tracking callbacks
    model.add_callback("on_train_epoch_begin", on_train_epoch_begin_perf)
    model.add_callback("on_train_batch_end", on_train_batch_end_perf)
    model.add_callback("on_train_epoch_end", on_train_epoch_end_perf)

    # Check if the model has GPU available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Train the YOLO model for 1 epoch and batch size 1 (adjust batch size for small dataset)
    model.train(
        data='data/yolo_format/noaug/data.yaml',
        epochs=1,  # Set to 1 epoch for quick testing
        batch=1,   # Adjust the batch size as needed (batch 1 for the small dataset)
        device=device
    )

    best = 'runs/train/exp/weights/best.pt'

    # Prepare evaluation mappings
    ptoi = {p:i for i,p in enumerate(ann['all_parts'])}
    itop = {i:p for p,i in ptoi.items()}
    all_idx = set(ptoi.values())

    # Evaluation
    def evaluate(split):
        results = []
        for img in splits[split].image_paths:
            det = YOLO(best).predict(source=str(img), conf=0.25)[0]
            preds = set(det.boxes.cls.cpu().numpy().astype(int).tolist())
            true = set(ptoi[p] for p in ann['images'][img.name]['missing_parts'])
            results.append({'predicted_missing_parts': all_idx - preds, 'true_missing_parts': true})
        part_level_evaluation(results, ptoi, itop, split)

    def part_level_evaluation(results, part_to_idx, idx_to_part, split_name):
        parts = list(part_to_idx.values())
        Yt = np.array([[1 if p in r['true_missing_parts'] else 0 for p in parts] for r in results])
        Yp = np.array([[1 if p in r['predicted_missing_parts'] else 0 for p in parts] for r in results])
        micro = f1_score(Yt, Yp, average='micro', zero_division=0)
        macro = f1_score(Yt, Yp, average='macro', zero_division=0)
        FN = (Yt == 1 & Yp == 0).sum()
        TP = (Yt == 1 & Yp == 1).sum()
        FP = (Yt == 0 & Yp == 1).sum()
        N = len(results)
        miss = FN / (FN + TP) if FN + TP > 0 else 0
        fppi = FP / N
        print(f"=== [METRICS] {split_name} ===")
        print(f"Micro-F1: {micro:.4f}, Macro-F1: {macro:.4f}, Miss: {miss:.4f}, FPPI: {fppi:.4f}")
        # per-part table
        table = []
        for i, p in enumerate(part_to_idx):
            acc = accuracy_score(Yt[:, i], Yp[:, i])
            prec = precision_score(Yt[:, i], Yp[:, i], zero_division=0)
            rec = recall_score(Yt[:, i], Yp[:, i], zero_division=0)
            f1 = f1_score(Yt[:, i], Yp[:, i], zero_division=0)
            table.append([idx_to_part[i], f"{acc:.3f}", f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}"])
        print(tabulate(table, headers=["Part", "Acc", "Prec", "Rec", "F1"], tablefmt="fancy_grid"))

    for s in ['val', 'test']:
        evaluate(s)
