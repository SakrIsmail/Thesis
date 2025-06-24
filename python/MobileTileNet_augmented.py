import os
import time
import json
import random
import numpy as np
from PIL import Image
from tabulate import tabulate
from tqdm import tqdm
from typing import List, Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import psutil
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlShutdown,
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from codecarbon import EmissionsTracker

TILE_SIZE = 80
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """
    Set random seed for reproducibility.
    Args:
        seed (int): Seed value to set for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

transform = transforms.Compose(
    [
        transforms.Resize((TILE_SIZE, TILE_SIZE)),
        transforms.ToTensor(),
    ]
)


def part_in_center_tile(part_bbox, image_size, tile_size=TILE_SIZE):
    """
    Check if a part's bounding box center is within the center tile of an image.
    Args:
        part_bbox (dict): Bounding box of the part with keys "left", "top",
                          "width", and "height".
        image_size (tuple): Size of the image as (width, height).
        tile_size (int): Size of the tile to check against.
    Returns:
        bool: True if the part's center is within the center tile, False otherwise.
    """
    img_w, img_h = image_size
    cx, cy = img_w // 2, img_h // 2
    tile_left = cx - tile_size // 2
    tile_top = cy - tile_size // 2
    tile_right = tile_left + tile_size
    tile_bottom = tile_top + tile_size

    part_cx = part_bbox["left"] + part_bbox["width"] / 2
    part_cy = part_bbox["top"] + part_bbox["height"] / 2

    return (tile_left <= part_cx <= tile_right) and (tile_top <= part_cy <= tile_bottom)


def find_anchor_part(images_subset, all_parts, image_dir):
    """
    Find the part that appears most frequently in the center tile of images.
    Args:
        images_subset (dict): Subset of images with their annotations.
        all_parts (list): List of all part names.
        image_dir (str): Directory where images are stored.
    Returns:
        str: The name of the part that appears most frequently in the center tile.
    """
    anchor_counts = {part: 0 for part in all_parts}
    for img_id, img_data in images_subset.items():
        img_path = os.path.join(image_dir, img_id)
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        for part in img_data.get("available_parts", []):
            name = part["part_name"]
            bbox = part["absolute_bounding_box"]
            center_x = bbox["left"] + bbox["width"] / 2
            center_y = bbox["top"] + bbox["height"] / 2
            left = img_w // 2 - TILE_SIZE // 2
            top = img_h // 2 - TILE_SIZE // 2
            if (
                left <= center_x <= left + TILE_SIZE
                and top <= center_y <= top + TILE_SIZE
            ):
                anchor_counts[name] += 1

    anchor_part = max(anchor_counts, key=anchor_counts.get)
    return anchor_part


def load_avg_positions_from_subset_with_anchor(
    train_images_subset, all_parts, anchor_part, image_dir
):
    """
    Load average positions of parts relative to the anchor part.
    Args:
        train_images_subset (dict): Subset of training images with annotations.
        all_parts (list): List of all part names.
        anchor_part (str): The name of the anchor part.
        image_dir (str): Directory where images are stored.
    Returns:
        Tuple: Average position of the anchor part and average offsets for all parts.
    """
    anchor_positions = []
    part_positions = {part: [] for part in all_parts}

    for img_id, img_data in train_images_subset.items():
        img_path = os.path.join(image_dir, img_id)
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        anchor_pos = None
        for part in img_data.get("available_parts", []):
            name = part["part_name"]
            bbox = part["absolute_bounding_box"]
            center_x = bbox["left"] + bbox["width"] / 2
            center_y = bbox["top"] + bbox["height"] / 2
            if name == anchor_part:
                anchor_pos = (center_x, center_y)
                break
        if anchor_pos is None:
            continue
        anchor_positions.append(anchor_pos)
        for part in img_data.get("available_parts", []):
            name = part["part_name"]
            bbox = part["absolute_bounding_box"]
            cx = bbox["left"] + bbox["width"] / 2
            cy = bbox["top"] + bbox["height"] / 2
            rel_x = cx - anchor_pos[0]
            rel_y = cy - anchor_pos[1]
            part_positions[name].append((rel_x, rel_y))

    avg_anchor_pos = np.mean(anchor_positions, axis=0)
    avg_offsets = {}
    for part in all_parts:
        if part_positions[part]:
            avg_offsets[part] = np.mean(part_positions[part], axis=0)
        else:
            avg_offsets[part] = (0, 0)
    return avg_anchor_pos, avg_offsets


def crop_tile(image: Image.Image, center: Tuple[int, int]) -> Image.Image:
    """
    Crop a tile of size TILE_SIZE around the given center coordinates.
    Args:
        image (Image.Image): The input image to crop from.
        center (Tuple[int, int]): The center coordinates (x, y) around which to crop.
    Returns:
        Image.Image: The cropped tile image.
    """
    cx, cy = center
    left = max(cx - TILE_SIZE // 2, 0)
    top = max(cy - TILE_SIZE // 2, 0)
    right = left + TILE_SIZE
    bottom = top + TILE_SIZE

    right = min(right, image.width)
    bottom = min(bottom, image.height)
    left = right - TILE_SIZE
    top = bottom - TILE_SIZE
    return image.crop((left, top, right, bottom))


def estimate_other_tiles(
    center: Tuple[int, int], all_parts: List[str], average_offsets: dict
):
    """
    Estimate the centers of other tiles based on the average offsets from the center tile.
    Args:
        center (Tuple[int, int]): The center coordinates (x, y) of the center tile.
        all_parts (List[str]): List of all part names.
        average_offsets (dict): Dictionary mapping part names to their average offsets.
    Returns:
        List[Tuple[int, int]]: List of estimated center coordinates for each part's tile.
    """
    cx, cy = center
    tile_centers = []
    for part in all_parts:
        dx, dy = average_offsets.get(part, (0, 0))
        est_x = int(cx + dx)
        est_y = int(cy + dy)
        tile_centers.append((est_x, est_y))
    return tile_centers


class BikeTileDataset(Dataset):
    """
    Dataset for loading bike tile images and their corresponding labels.
    This dataset crops tiles around estimated centers based on the average offsets
    from a specified anchor part, and returns the tiles along with labels indicating
    the presence or absence of parts in the center tile.
    """

    def __init__(
        self,
        annotations,
        image_dir,
        image_ids,
        all_parts,
        average_offsets,
        transform=None,
        augment=True,
        target_size=(640, 640),
    ):
        self.image_dir = image_dir
        self.images = image_ids
        self.annotations = annotations["images"]
        self.part_to_idx = {part: i for i, part in enumerate(all_parts)}
        self.all_parts = all_parts
        self.average_offsets = average_offsets
        self.target_size = target_size
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.images)

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
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        original_w, original_h = image.size
        annotation = self.annotations[img_name]
        parts = annotation.get("available_parts", [])

        # Convert bounding boxes to [xmin, ymin, xmax, ymax] format
        boxes = []
        for part in parts:
            bbox = part["absolute_bounding_box"]
            xmin = bbox["left"]
            ymin = bbox["top"]
            xmax = xmin + bbox["width"]
            ymax = ymin + bbox["height"]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.tensor(boxes, dtype=torch.float32)

        # Apply augmentations before resizing
        if self.augment:
            image, boxes = self.apply_augmentation(image, boxes)

        # Resize image and scale boxes
        image = image.resize(self.target_size, Image.BILINEAR)
        resized_w, resized_h = self.target_size
        scale_x = resized_w / original_w
        scale_y = resized_h / original_h
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # Update parts with scaled bbox values
        for i, part in enumerate(parts):
            part["absolute_bounding_box"] = {
                "left": boxes[i][0].item(),
                "top": boxes[i][1].item(),
                "width": (boxes[i][2] - boxes[i][0]).item(),
                "height": (boxes[i][3] - boxes[i][1]).item(),
            }

        # Estimate tile centers
        cx, cy = resized_w // 2, resized_h // 2
        tile_centers = estimate_other_tiles(
            (cx, cy), self.all_parts, self.average_offsets
        )

        # Crop and transform tiles
        tiles = [crop_tile(image, center) for center in tile_centers]
        if self.transform:
            tiles = [self.transform(tile) for tile in tiles]
        else:
            tiles = [transforms.functional.to_tensor(tile) for tile in tiles]

        tiles = torch.stack(tiles)

        # Create label vector
        missing_parts = annotation.get("missing_parts", [])
        label = torch.zeros(len(self.all_parts))
        for part in missing_parts:
            idx_part = self.part_to_idx[part]
            label[idx_part] = 1

        return tiles, label


class TileMobileNet(nn.Module):
    """
    MobileNet model for processing bike tiles.
    This model uses a MobileNetV2 backbone to extract features from the input tiles,
    applies adaptive average pooling, and then classifies the aggregated features into
    the specified number of parts.
    """

    def __init__(self, num_parts=22):
        super().__init__()
        backbone = models.mobilenet_v2(pretrained=True)
        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, num_parts),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x)
        pooled = self.pool(feats).view(B, T, -1)
        aggregated = pooled.mean(dim=1)
        out = self.classifier(aggregated)
        return out


if torch.cuda.is_available():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)


def train(model, dataloader, optimizer, criterion):
    """Train the model for one epoch.
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function to compute the loss.
    """
    model.train()
    total_loss, total_time, total_pixels = 0, 0, 0
    global gpu_memories, cpu_memories, batch_times
    batch_times = []
    gpu_memories = []
    cpu_memories = []
    
    for tiles, labels in tqdm(dataloader):
        tiles = tiles.to(DEVICE)
        labels = labels.to(DEVICE)

        start_time = time.time()

        preds = model(tiles)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        total_loss += loss.item()
        total_time += batch_time
        total_pixels += tiles.numel()

        if torch.cuda.is_available():
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            gpu_mem_used = mem_info.used / (1024**2)
            gpu_memories.append(gpu_mem_used)
        else:
            gpu_mem_used = 0

        cpu_mem_used = psutil.virtual_memory().used / (1024**2)
        cpu_memories.append(cpu_mem_used)

    full_img_pixels = 640 * 640 * 3
    total_full_image_pixels = full_img_pixels * len(dataloader.dataset)
    pixels_saved = 100 * (1 - total_pixels / total_full_image_pixels)
    avg_loss = total_loss / len(dataloader)
    print(
        f"Train Loss: {avg_loss:.4f}, Time: {total_time:.2f}s, Saved pixels: {pixels_saved:.2f}%"
    )


def evaluate(model, dataloader, criterion):
    """
    Evaluate the model on the validation or test dataset.
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the validation or test dataset.
        criterion (nn.Module): Loss function to compute the loss.
    """
    model.eval()
    total_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for tiles, labels in tqdm(dataloader):
            tiles = tiles.to(DEVICE)
            labels = labels.to(DEVICE)
            preds = model(tiles)
            loss = criterion(preds, labels)
            total_loss += loss.item()

            preds_bin = (torch.sigmoid(preds) > 0.5).cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.append(preds_bin)
            all_labels.append(labels_np)

    avg_loss = total_loss / len(dataloader)
    print(f"Loss: {avg_loss:.4f}")

    Y_pred = np.vstack(all_preds)
    Y_true = np.vstack(all_labels)

    micro_f1 = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(Y_true, Y_pred, average="macro", zero_division=0)

    FN = np.logical_and(Y_true == 1, Y_pred == 0).sum()
    TP = np.logical_and(Y_true == 1, Y_pred == 1).sum()
    FP = np.logical_and(Y_true == 0, Y_pred == 1).sum()

    N_images = Y_true.shape[0]
    miss_rate = FN / (FN + TP) if (FN + TP) > 0 else 0
    fppi = FP / N_images

    overall_acc = accuracy_score(Y_true.flatten(), Y_pred.flatten())
    overall_prec = precision_score(Y_true.flatten(), Y_pred.flatten(), zero_division=0)
    overall_rec = recall_score(Y_true.flatten(), Y_pred.flatten(), zero_division=0)
    overall_f1 = f1_score(Y_true.flatten(), Y_pred.flatten(), zero_division=0)

    print(f"Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}")
    print(f"Miss Rate: {miss_rate:.4f}, FPPI: {fppi:.4f}")
    print(
        f"Overall Acc: {overall_acc:.4f}, Precision: {overall_prec:.4f}, Recall: {overall_rec:.4f}, F1: {overall_f1:.4f}"
    )


if __name__ == "__main__":
    json_path = (
        "/var/scratch/sismail/data/processed/final_annotations_without_occluded.json"
    )
    image_dir = "/var/scratch/sismail/data/images"

    with open(json_path) as f:
        annotations = json.load(f)

    image_ids = list(annotations["images"].keys())
    random.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(0.8 * n)
    n_test = n - n_train
    n_val = int(0.1 * n_train)
    n_train = n_train - n_val

    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train : n_train + n_val]
    test_ids = image_ids[n_train + n_val :]

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    train_images_subset = {k: annotations["images"][k] for k in train_ids}
    all_parts = annotations["all_parts"]

    anchor_part = find_anchor_part(train_images_subset, all_parts, image_dir)
    avg_anchor_pos, average_offsets = load_avg_positions_from_subset_with_anchor(
        train_images_subset, all_parts, anchor_part, image_dir
    )

    print(f"Anchor part: {anchor_part}")
    print(f"Average anchor position: {avg_anchor_pos}")
    print(
        f"Average offsets for parts (example): {dict(list(average_offsets.items())[:3])}"
    )

    train_dataset = BikeTileDataset(
        annotations, image_dir, train_ids, all_parts, average_offsets, augment=True
    )
    val_dataset = BikeTileDataset(
        annotations, image_dir, val_ids, all_parts, average_offsets, augment=False
    )
    test_dataset = BikeTileDataset(
        annotations, image_dir, test_ids, all_parts, average_offsets, augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = TileMobileNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )

    epochs = 100
    patience = 8
    best_macro_f1 = 0
    no_improve = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        with EmissionsTracker(log_level="critical", save_to_file=False) as tracker:
            train(model, train_loader, optimizer, criterion)
            model.eval()
            total_loss = 0

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for tiles, labels in tqdm(val_loader):
                    tiles = tiles.to(DEVICE)
                    labels = labels.to(DEVICE)
                    preds = model(tiles)
                    loss = criterion(preds, labels)
                    total_loss += loss.item()

                    preds_bin = (torch.sigmoid(preds) > 0.5).cpu().numpy()
                    labels_np = labels.cpu().numpy()

                    all_preds.append(preds_bin)
                    all_labels.append(labels_np)

            avg_loss = total_loss / len(val_loader)
            print(f"Loss: {avg_loss:.4f}")

            Y_pred = np.vstack(all_preds)
            Y_true = np.vstack(all_labels)

            macro_f1 = f1_score(Y_true, Y_pred, average="macro", zero_division=0)

            sched.step(macro_f1)

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                no_improve = 0
                torch.save(
                    model.state_dict(),
                    "/var/scratch/sismail/models/MobileTileNet_augmented_model.pth",
                )
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        energy_consumption = tracker.final_emissions_data.energy_consumed
        co2_emissions = tracker.final_emissions

        avg_time = np.mean(batch_times)
        max_gpu_mem = max(gpu_memories) if gpu_memories else 0
        max_cpu_mem = max(cpu_memories)

        table = [
            ["Epoch", f"{epoch+1:.2f}"],
            ["Average Batch Time (s)", f"{avg_time:.4f}"],
            ["Maximum GPU Memory Usage (MB)", f"{max_gpu_mem:.2f}"],
            ["Maximum CPU Memory Usage (MB)", f"{max_cpu_mem:.2f}"],
            ["Energy Consumption (kWh)", f"{energy_consumption:.4f} kWh"],
            ["CO₂ Emissions (kg)", f"{co2_emissions:.4f} kg"],
        ]

        print(tabulate(table, headers=["Metric", "Value"], tablefmt="pretty"))

    print("Training complete.")
    if torch.cuda.is_available():
        nvmlShutdown()
    model.load_state_dict(
        torch.load("/var/scratch/sismail/models/MobileTileNet_augmented_model.pth")
    )
    print("Evaluating on validation set:")
    evaluate(model, val_loader, criterion)

    print("\nFinal evaluation on test set:")
    evaluate(model, test_loader, criterion)
