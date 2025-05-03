#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import random
from tqdm import tqdm
import time
import psutil
import gc
from tabulate import tabulate
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown



# In[ ]:


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



# In[12]:


class BikePartsDetectionDataset(Dataset):
    def __init__(self, annotations_dict, image_dir, transform=None, augment=True, target_size=(640, 640)):
        self.all_parts = annotations_dict['all_parts']
        self.part_to_idx = {part: idx + 1 for idx, part in enumerate(self.all_parts)}
        self.idx_to_part = {idx + 1: part for idx, part in enumerate(self.all_parts)}
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



# In[13]:

train_dataset = BikePartsDetectionDataset(
    annotations_dict=train_annotations,
    image_dir=image_directory,
    augment=True
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
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch))
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch))
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch))
)


# In[ ]:


model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')

in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = len(train_dataset.all_parts) + 1
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

if torch.cuda.is_available():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()

    batch_times = []
    gpu_memories = []
    cpu_memories = []

    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs}") as tepoch:
        for images, targets in tepoch:
            start_time = time.time()

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
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
                "loss": f"{losses.item():.4f}",
                "time (s)": f"{inference_time:.3f}",
                "GPU Mem (MB)": f"{gpu_mem_used:.0f}",
                "CPU Mem (MB)": f"{cpu_mem_used:.0f}"
            })

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_time = sum(batch_times) / len(batch_times)
    max_gpu_mem = max(gpu_memories)
    max_cpu_mem = max(cpu_memories)

    table = [
        ["Epoch", epoch + 1],
        ["Final Loss", f"{losses.item():.4f}"],
        ["Average Batch Time (sec)", f"{avg_time:.4f}"],
        ["Average GPU Memory Usage (MB)", f"{max_gpu_mem:.2f}"],
        ["Average CPU Memory Usage (MB)", f"{max_cpu_mem:.2f}"],
    ]

    print(tabulate(table, headers=["Metric", "Value"], tablefmt="pretty"))


if torch.cuda.is_available():
    nvmlShutdown()

torch.save(model.state_dict(), f"/var/scratch/sismail/models/faster_rcnn/fasterrcnn_MobileNet_stretched_{num_epochs}_model.pth")


# In[ ]:


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
            pred_parts = set(predictions[i]['labels'].cpu().numpy().tolist())
            true_missing_parts = set(targets[i]['missing_labels'].cpu().numpy().tolist())
            image_id = targets[i]['image_id'].item()

            predicted_missing_parts = all_parts_set - pred_parts

            results_per_image.append({
                'image_id': image_id,
                'predicted_missing_parts': predicted_missing_parts,
                'true_missing_parts': true_missing_parts
            })

    return results_per_image


def part_level_evaluation(results_per_image, part_to_idx, idx_to_part):
    part_indices = list(part_to_idx.values())

    part_true = {part: [] for part in part_indices}
    part_pred = {part: [] for part in part_indices}

    for result in results_per_image:
        true_missing = result['true_missing_parts']
        predicted_missing = result['predicted_missing_parts']

        for part in part_indices:
            part_true[part].append(1 if part in true_missing else 0)
            part_pred[part].append(1 if part in predicted_missing else 0)

    accuracy = {}
    precision = {}
    recall = {}
    f1 = {}

    all_true_flat = []
    all_pred_flat = []

    table_rows = []

    for part in part_indices:
        y_true = part_true[part]
        y_pred = part_pred[part]

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1s = f1_score(y_true, y_pred, zero_division=0)

        accuracy[part] = acc
        precision[part] = prec
        recall[part] = rec
        f1[part] = f1s

        all_true_flat.extend(y_true)
        all_pred_flat.extend(y_pred)

        table_rows.append([
            idx_to_part[part], f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1s:.4f}"
        ])

    print(tabulate(table_rows, headers=["Part", "Accuracy", "Precision", "Recall", "F1 Score"], tablefmt="fancy_grid"))

    overall_accuracy = accuracy_score(all_true_flat, all_pred_flat)
    overall_precision = precision_score(all_true_flat, all_pred_flat, zero_division=0)
    overall_recall = recall_score(all_true_flat, all_pred_flat, zero_division=0)
    overall_f1 = f1_score(all_true_flat, all_pred_flat, zero_division=0)

    print("\nOverall Metrics:")
    print(tabulate([[
        f"{overall_accuracy:.4f}", f"{overall_precision:.4f}",
        f"{overall_recall:.4f}", f"{overall_f1:.4f}"
    ]], headers=["Accuracy", "Precision", "Recall", "F1 Score"], tablefmt="fancy_grid"))

    return accuracy, precision, recall, f1, overall_accuracy, overall_precision, overall_recall, overall_f1


results_per_image = evaluate_model(model, valid_loader, train_dataset.part_to_idx, device)
# results_per_image = evaluate_model(model, test_loader, train_dataset.part_to_idx, device)

accuracy, precision, recall, f1, overall_accuracy, overall_precision, overall_recall, overall_f1 = part_level_evaluation(
    results_per_image, train_dataset.part_to_idx, train_dataset.idx_to_part
)