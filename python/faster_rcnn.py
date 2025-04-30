#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import random
from tqdm import tqdm
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


# In[ ]:


final_output_json='/var/scratch/sismail/data/processed/final_annotations.json'
image_directory = '/var/scratch/sismail/data/images'

test_ratio = 0.2
random_seed = 42

with open(final_output_json, 'r') as f:
    annotations = json.load(f)

image_filenames = list(annotations['images'].keys())[:500]

random.seed(random_seed)
random.shuffle(image_filenames)

num_test = int(len(image_filenames) * test_ratio)
test_images = image_filenames[:num_test]
train_images = image_filenames[num_test:]

train_annotations = {
    'all_parts': annotations['all_parts'],
    'images': {img_name: annotations['images'][img_name] for img_name in train_images}
}

test_annotations = {
    'all_parts': annotations['all_parts'],
    'images': {img_name: annotations['images'][img_name] for img_name in test_images}
}


# In[12]:


class BikePartsDetectionDataset(Dataset):
    def __init__(self, annotations_dict, image_dir, transform=None):
        self.all_parts = annotations_dict['all_parts']
        self.part_to_idx = {part: idx + 1 for idx, part in enumerate(self.all_parts)}
        self.idx_to_part = {idx + 1: part for idx, part in enumerate(self.all_parts)}
        self.image_data = annotations_dict['images']
        self.image_filenames = list(self.image_data.keys())
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_filename)

        image = Image.open(img_path).convert('RGB')

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

        missing_labels = torch.tensor([self.part_to_idx[part] for part in missing_parts_names], dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'missing_labels': missing_labels,
            'image_id': torch.tensor([idx])
        }

        if self.transform:
            image = self.transform(image)

        return image, target


# In[13]:


transform = transforms.ToTensor()

train_dataset = BikePartsDetectionDataset(train_annotations, image_directory, transform=transform)
test_dataset = BikePartsDetectionDataset(test_annotations, image_directory, transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
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

num_epochs = 1
for epoch in range(num_epochs):
    model.train()

    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs}") as tepoch:
        for images, targets in tepoch:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            tepoch.set_postfix(loss=losses.item())

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {losses.item()}")

torch.save(model.state_dict(), f"/var/scratch/sismail/models/faster_rcnn/fasterrcnn_{num_epochs}_model.pth")


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


def part_level_evaluation_sklearn(results_per_image, part_to_idx, idx_to_part):
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

    for part in part_indices:
        y_true = part_true[part]
        y_pred = part_pred[part]

        accuracy[part] = accuracy_score(y_true, y_pred)
        precision[part] = precision_score(y_true, y_pred, zero_division=0)
        recall[part] = recall_score(y_true, y_pred, zero_division=0)
        f1[part] = f1_score(y_true, y_pred, zero_division=0)

        all_true_flat.extend(y_true)
        all_pred_flat.extend(y_pred)

        print(f"Part: {idx_to_part[part]}")
        print(f"  Accuracy: {accuracy[part]:.4f}")
        print(f"  Precision: {precision[part]:.4f}")
        print(f"  Recall: {recall[part]:.4f}")
        print(f"  F1 Score: {f1[part]:.4f}\n")

    overall_accuracy = accuracy_score(all_true_flat, all_pred_flat)
    overall_precision = precision_score(all_true_flat, all_pred_flat, zero_division=0)
    overall_recall = recall_score(all_true_flat, all_pred_flat, zero_division=0)
    overall_f1 = f1_score(all_true_flat, all_pred_flat, zero_division=0)

    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")

    return accuracy, precision, recall, f1, overall_accuracy, overall_precision, overall_recall, overall_f1


results_per_image = evaluate_model(model, test_loader, train_dataset.part_to_idx, device)

accuracy, precision, recall, f1, overall_accuracy, overall_precision, overall_recall, overall_f1 = part_level_evaluation_sklearn(
    results_per_image, train_dataset.part_to_idx, train_dataset.idx_to_part
)

