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
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
from codecarbon import EmissionsTracker

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


final_output_json='/var/scratch/sismail/data/processed/final_direct_missing.json'
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
    def __init__(self, annotations_dict, image_dir,
                 transform=None, augment=True, target_size=(640,640)):
        self.all_parts     = annotations_dict['all_parts']
        self.part_to_idx   = {p:i+1 for i,p in enumerate(self.all_parts)}  # background=0
        self.idx_to_part   = {i+1:p for i,p in enumerate(self.all_parts)}
        self.image_data    = annotations_dict['images']
        self.image_filenames = list(self.image_data.keys())
        self.image_dir     = image_dir
        self.transform     = transform
        self.augment       = augment
        self.target_size   = target_size

    def __len__(self):
        # if augment, you double-sample half with flips/color jits
        return len(self.image_filenames) * (2 if self.augment else 1)

    def apply_augmentation(self, image, boxes):
        # same as yours
        if random.random() < 0.5:
            image = transforms.functional.hflip(image)
            w = image.width
            boxes = boxes.clone()
            boxes[:, [0,2]] = w - boxes[:, [2,0]]
        if random.random() < 0.8:
            image = transforms.functional.adjust_brightness(
                         image, brightness_factor=random.uniform(0.6,1.4))
        if random.random() < 0.8:
            image = transforms.functional.adjust_contrast(
                         image, contrast_factor=random.uniform(0.6,1.4))
        if random.random() < 0.5:
            image = transforms.functional.adjust_saturation(
                         image, saturation_factor=random.uniform(0.7,1.3))
        return image, boxes

    def __getitem__(self, idx):
        real_idx = idx % len(self.image_filenames)
        do_aug   = self.augment and (idx >= len(self.image_filenames))

        fn = self.image_filenames[real_idx]
        img = Image.open(os.path.join(self.image_dir,fn)).convert('RGB')
        ow, oh = img.size

        info = self.image_data[fn]['parts']
        boxes, labels, is_missing = [], [], []

        for part in info:
            bb = part['absolute_bounding_box']
            x0,y0 = bb['left'], bb['top']
            x1,y1 = x0+bb['width'], y0+bb['height']
            boxes.append([x0,y0,x1,y1])
            labels.append(self.part_to_idx[part['part_name']])
            is_missing.append(0 if part['present'] else 1)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        is_missing = torch.tensor(is_missing, dtype=torch.int64)

        if do_aug:
            img, boxes = self.apply_augmentation(img, boxes)

        # resize + rescale boxes
        img = transforms.functional.resize(img, self.target_size)
        sx, sy = self.target_size[0]/ow, self.target_size[1]/oh
        boxes[:,[0,2]] *= sx
        boxes[:,[1,3]] *= sy
        img = transforms.functional.to_tensor(img)

        target = {
            'boxes': boxes,
            'labels': labels,
            'is_missing': is_missing,
            'image_id': torch.tensor([real_idx])
        }
        return img, target



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

def visualize_and_save_predictions(model, dataset, device, out_dir="output_preds", n_images=5):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    for idx in range(min(n_images, len(dataset))):
        img, target = dataset[idx]
        # send through model
        with torch.no_grad():
            pred = model([img.to(device)])[0]

        # convert back to PIL for plotting
        img_np = img.mul(255).permute(1,2,0).byte().cpu().numpy()
        fig, ax = plt.subplots(1, figsize=(8,8))
        ax.imshow(img_np)

        # ground-truth missing boxes in red
        gt_boxes = target['boxes'][target['is_missing']==1].cpu().numpy()
        for (x0,y0,x1,y1) in gt_boxes:
            rect = patches.Rectangle((x0,y0), x1-x0, y1-y0,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # predicted missing boxes in blue
        if 'missing_flag' in pred:
            # for FasterRCNN variant that uses 'missing_flag'
            mask = pred['missing_flag']==1
        else:
            # for your 22-way head: any box where missing_preds[argmax]==1
            flags = pred['missing_preds'].argmax(dim=1)==1
            mask = flags.cpu()
        pred_boxes = pred['boxes'][mask].cpu().numpy()
        for (x0,y0,x1,y1) in pred_boxes:
            rect = patches.Rectangle((x0,y0), x1-x0, y1-y0,
                                     linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

        ax.axis('off')
        fname = os.path.join(out_dir, f"pred_{idx}.png")
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

def evaluate_model(model, loader, device):
    model.eval()
    results = []
    for imgs, targets in loader:
        imgs = [i.to(device) for i in imgs]
        with torch.no_grad():
            preds = model(imgs)

        for tgt, pred in zip(targets, preds):
            # full 22-vector: 1 if that part was flagged missing
            vec = torch.zeros(len(model.backbone_model.roi_heads.box_predictor.cls_score.weight), dtype=torch.int64)
            for lbl,flag in zip(pred['labels'].cpu(), pred['missing_flag']):
                if flag.item()==1:
                    vec[lbl-1] = 1
            # ground truth:
            true_vec = torch.zeros_like(vec)
            for lbl, m in zip(tgt['labels'], tgt['is_missing']):
                true_vec[lbl-1] = m

            results.append({
              'image_id': tgt['image_id'].item(),
              'predicted_missing_parts': vec.tolist(),
              'true_missing_parts':       true_vec.tolist()
            })
    return results



def part_level_evaluation(results, all_parts):
    """
    results: list of dicts with keys
        - 'predicted_missing_parts': List[int] of length P (0/1 for each part)
        - 'true_missing_parts':      List[int] of length P (0/1 for each part)
    all_parts:  List[str] of length P, giving the part names in order
    """
    # Stack into (N_images, P) arrays
    Y_true = np.array([r['true_missing_parts']      for r in results])
    Y_pred = np.array([r['predicted_missing_parts'] for r in results])

    # Overall (flattened) metrics
    micro_f1 = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(Y_true, Y_pred, average='macro', zero_division=0)
    overall_acc = accuracy_score(Y_true.flatten(), Y_pred.flatten())
    overall_prec = precision_score(Y_true.flatten(), Y_pred.flatten(), zero_division=0)
    overall_rec  = recall_score(Y_true.flatten(), Y_pred.flatten(), zero_division=0)
    overall_f1   = f1_score(Y_true.flatten(), Y_pred.flatten(), zero_division=0)

    # Miss-rate and FPPI
    FN = np.logical_and(Y_true==1, Y_pred==0).sum()
    TP = np.logical_and(Y_true==1, Y_pred==1).sum()
    FP = np.logical_and(Y_true==0, Y_pred==1).sum()
    miss_rate = FN/(FN+TP) if (FN+TP)>0 else 0.0
    fppi = FP / len(results)

    print(f"[METRIC] Micro-F1: {micro_f1:.4f}")
    print(f"[METRIC] Macro-F1: {macro_f1:.4f}")
    print(f"[METRIC] Miss Rate: {miss_rate:.4f}")
    print(f"[METRIC] FPPI: {fppi:.4f}")
    print(f"[METRIC] Overall Acc: {overall_acc:.4f}")
    print(f"[METRIC] Precision: {overall_prec:.4f}")
    print(f"[METRIC] Recall: {overall_rec:.4f}")
    print(f"[METRIC] F1: {overall_f1:.4f}")

    # Per-part table
    table = []
    for j, part_name in enumerate(all_parts):
        acc  = accuracy_score(Y_true[:,j], Y_pred[:,j])
        prec = precision_score(Y_true[:,j], Y_pred[:,j], zero_division=0)
        rec  = recall_score(Y_true[:,j], Y_pred[:,j], zero_division=0)
        f1   = f1_score(Y_true[:,j], Y_pred[:,j], zero_division=0)
        table.append([part_name, f"{acc:.3f}", f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}"])

    print("\n[PER-PART EVALUATION]")
    print(tabulate(table, headers=["Part","Acc","Prec","Rec","F1"], tablefmt="fancy_grid"))

class BikePartsFasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # load base Faster-RCNN
        self.backbone_model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
        in_f = self.backbone_model.roi_heads.box_predictor.cls_score.in_features
        # replace the detection head
        self.backbone_model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
        # add a missing/not‐missing head on top of the RoI features
        self.missing_head = nn.Linear(in_f, 2)

    def forward(self, images, targets=None):
        if self.training:
            # 1) standard detection losses
            losses = self.backbone_model(images, targets)

            # 2) extract feature‐maps
            img_tensor = torch.stack(images)
            features   = self.backbone_model.backbone(img_tensor)

            # 3) gather all GT boxes & missing flags
            all_boxes   = [t['boxes'] for t in targets]
            all_labels  = [t['is_missing'] for t in targets]
            shapes      = [img.shape[-2:] for img in images]

            # skip empty images
            valid = [i for i,b in enumerate(all_boxes) if b.numel()>0]
            if valid:
                vb   = [all_boxes[i]  for i in valid]
                vm   = torch.cat([all_labels[i] for i in valid], dim=0)
                vsz  = [shapes[i]     for i in valid]

                # RoI‐align + box‐head to get per‐RoI features
                rois = self.backbone_model.roi_heads.box_roi_pool(features, vb, vsz)
                head_feats = self.backbone_model.roi_heads.box_head(rois)
                # missing‐loss
                logits = self.missing_head(head_feats)
                losses['loss_missing'] = nn.functional.cross_entropy(logits, vm)

            return losses

        else:
            # inference: get raw detections
            outputs = self.backbone_model(images)

            # extract features once
            img_tensor = torch.stack(images)
            features   = self.backbone_model.backbone(img_tensor)

            for img_idx, out in enumerate(outputs):
                boxes = out['boxes']
                if boxes.numel()==0:
                    out['missing_flag'] = torch.zeros((0,), dtype=torch.int64)
                    continue

                # RoI + head
                rois  = self.backbone_model.roi_heads.box_roi_pool(
                             features, [boxes], [images[img_idx].shape[-2:]])
                feats = self.backbone_model.roi_heads.box_head(rois)
                logits= self.missing_head(feats)
                # pick argmax over the 2 classes
                flags = logits.argmax(dim=1).cpu()
                out['missing_flag'] = flags

                # if you only want the missing boxes:
                out['boxes_missing'] = boxes[flags==1]
                out['labels_missing']= out['labels'][flags==1]

            return outputs
        

num_classes = len(train_dataset.all_parts) + 1 
model = BikePartsFasterRCNN(num_classes=num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

learning_rate = 1e-4
weight_decay = 1e-4

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

if torch.cuda.is_available():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

num_epochs = 20

for epoch in range(num_epochs):
    with EmissionsTracker(log_level="critical", save_to_file=False) as tracker:

        model.train()

        batch_times = []
        gpu_memories = []
        cpu_memories = []

        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs}") as tepoch:
            for images, targets in tepoch:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                start_time = time.time()

                optimizer.zero_grad()
                loss_dict = model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())
                total_loss.backward()
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
                    "loss": f"{total_loss.item():.4f}",
                    "time (s)": f"{inference_time:.3f}",
                    "GPU Mem (MB)": f"{gpu_mem_used:.0f}",
                    "CPU Mem (MB)": f"{cpu_mem_used:.0f}"
                })

                del loss_dict, images, targets
                gc.collect()
                if torch.cuda.is_available(): 
                    torch.cuda.empty_cache()


    energy_consumption = tracker.final_emissions_data.energy_consumed
    co2_emissions = tracker.final_emissions

    avg_time = sum(batch_times) / len(batch_times)
    max_gpu_mem = max(gpu_memories) if gpu_memories else 0
    max_cpu_mem = max(cpu_memories)

    table = [
        ["Epoch", epoch + 1],
        ["Final Loss", f"{total_loss.item():.4f}"],
        ["Average Batch Time (sec)", f"{avg_time:.4f}"],
        ["Maximum GPU Memory Usage (MB)", f"{max_gpu_mem:.2f}"],
        ["Maximum CPU Memory Usage (MB)", f"{max_cpu_mem:.2f}"],    
        ["Energy Consumption (kWh)", f"{energy_consumption:.4f} kWh"],
        ["CO₂ Emissions (kg)", f"{co2_emissions:.4f} kg"],
    ]

    print(tabulate(table, headers=["Metric", "Value"], tablefmt="pretty"))

torch.save(model.state_dict(), "/var/scratch/sismail/models/faster_rcnn/fasterrcnn_MobileNet_missing_baseline_model.pth")

if torch.cuda.is_available():
    nvmlShutdown()


model.load_state_dict(torch.load("/var/scratch/sismail/models/faster_rcnn/fasterrcnn_MobileNet_missing_baseline_model.pth", map_location=device))
model.to(device)

model.eval()

results_per_image = evaluate_model(model, valid_loader, device)
print(results_per_image)

part_level_evaluation(results_per_image, train_dataset.all_parts)

visualize_and_save_predictions(model, valid_dataset, device, out_dir="/home/sismail/Thesis/visualisations/", n_images=5)


results_per_image = evaluate_model(model, test_loader, device)

part_level_evaluation(results_per_image, train_dataset.all_parts)
