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
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
from codecarbon import EmissionsTracker
from sklearn.model_selection import ParameterGrid



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
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch))
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch))
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch))
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
    print(f"Micro-F1: {micro_f1:.4f}, Macro-F1: {macro_f1:.4f}")
    print(f"Miss Rate: {miss_rate:.4f}, FPPI: {fppi:.4f}")
    print(f"Overall Acc: {overall_acc:.4f}, Precision: {overall_prec:.4f}, Recall: {overall_rec:.4f}, F1: {overall_f1:.4f}")
    
    table=[]
    for j,p in enumerate(parts):
        acc = accuracy_score(Y_true[:,j], Y_pred[:,j])
        prec = precision_score(Y_true[:,j], Y_pred[:,j], zero_division=0)
        rec = recall_score(Y_true[:,j], Y_pred[:,j], zero_division=0)
        f1s = f1_score(Y_true[:,j], Y_pred[:,j], zero_division=0)
        table.append([idx_to_part[p], f"{acc:.3f}", f"{prec:.3f}", f"{rec:.3f}", f"{f1s:.3f}"])
    print(tabulate(table, headers=["Part","Acc","Prec","Rec","F1"], tablefmt="fancy_grid"))



model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')

in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = len(train_dataset.all_parts) + 1
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

param_grid = {
    'learning_rate': [1e-4, 1e-5, 1e-6],
    'weight_decay': [1e-4, 1e-5, 1e-6],
}

grid = ParameterGrid(param_grid)

best_model = None
best_f1_score = 0
best_params = None

for params in grid:
    print(f"Training with params: {params}")
    
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

        results_per_image = evaluate_model(model, valid_loader, train_dataset.part_to_idx, device)

        parts = list(train_dataset.part_to_idx.values())
        Y_true = np.array([[1 if p in r['true_missing_parts'] else 0 for p in parts] for r in results_per_image])
        Y_pred = np.array([[1 if p in r['predicted_missing_parts'] else 0 for p in parts] for r in results_per_image])

        micro_f1 = f1_score(Y_true.flatten(), Y_pred.flatten(), average='micro')
        
        if micro_f1 > best_f1_score:
            best_f1_score = micro_f1
            best_model = model
            best_params = params

    print(f"Best F1 Score: {best_f1_score:.4f} with params: {best_params}")