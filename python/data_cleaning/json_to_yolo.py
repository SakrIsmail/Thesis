import os
import numpy as np
import torch
import json
import random
import shutil
from PIL import Image
from torchvision import transforms
import yaml as _yaml


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


def prepare_splits(
    annotation_json, test_ratio=0.2, valid_ratio=0.1, seed=42, limit=None
):
    """
    Prepare train, validation, and test splits from the given annotation JSON file.
    Args:
        annotation_json (str): Path to the JSON file containing annotations.
        test_ratio (float): Proportion of the dataset to include in the test split.
        valid_ratio (float): Proportion of the training set to include in the validation split.
        seed (int): Random seed for reproducibility.
        limit (int, optional): Limit the number of images to process. If None, process all images.
    Returns:
        dict: A dictionary containing lists of image filenames for 'train', 'val', and 'test' splits.
        dict: The original annotations loaded from the JSON file.
    """
    with open(annotation_json, "r") as f:
        annotations = json.load(f)
    image_filenames = list(annotations["images"].keys())
    if limit:
        image_filenames = image_filenames[:limit]
    random.seed(seed)
    random.shuffle(image_filenames)
    n = len(image_filenames)
    n_test = int(n * test_ratio)
    n_train = n - n_test
    n_val = int(n_train * valid_ratio)
    test = image_filenames[:n_test]
    val = image_filenames[n_test : n_test + n_val]
    train = image_filenames[n_test + n_val :]
    return {"train": train, "val": val, "test": test}, annotations


def apply_augmentation(image, boxes):
    """
    Apply random augmentations to the image and bounding boxes.
    Args:
        image (PIL.Image): The input image.
        boxes (torch.Tensor): Bounding boxes in the format [x_min, y_min, x_max, y_max].
    Returns:
        PIL.Image: Augmented image.
        torch.Tensor: Augmented bounding boxes.
    """

    boxes = torch.tensor(boxes, dtype=torch.float32)

    if random.random() < 0.5:
        image = transforms.functional.hflip(image)
        w = image.width
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


def convert_json_to_yolo(
    annotation_json, image_dir, base_out_dir, splits, augment=False
):
    """
    Convert annotations from a JSON file to YOLO format and save images and labels.
    Args:
        annotation_json (str): Path to the JSON file containing annotations.
        image_dir (str): Directory containing the images.
        base_out_dir (str): Base output directory where the YOLO formatted data will be saved.
        splits (dict): Dictionary containing lists of image filenames for 'train', 'val', and 'test' splits.
        augment (bool): Whether to apply data augmentation to the training images.
    """
    with open(annotation_json, "r") as f:
        annotations = json.load(f)
    all_parts = annotations["all_parts"]
    part_to_idx = {p: i for i, p in enumerate(all_parts)}

    target_size = (640, 640)

    for split, filenames in splits.items():
        img_out = os.path.join(base_out_dir, "images", split)
        lbl_out = os.path.join(base_out_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for fn in filenames:
            src_img = os.path.join(image_dir, fn)
            img = Image.open(src_img).convert("RGB")
            orig_w, orig_h = img.size

            scale_x = target_size[0] / orig_w
            scale_y = target_size[1] / orig_h

            parts = annotations["images"][fn]["available_parts"]
            boxes = []
            labels = []

            for part in parts:
                cls = part_to_idx[part["part_name"]]
                bb = part["absolute_bounding_box"]
                xmin = bb["left"]
                ymin = bb["top"]
                xmax = xmin + bb["width"]
                ymax = ymin + bb["height"]
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cls)

            # Resize original image
            img_resized = img.resize(target_size, Image.BILINEAR)
            img_resized.save(os.path.join(img_out, fn))

            # Scale boxes
            boxes = [
                [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
                for b in boxes
            ]

            # Write YOLO label
            lines = []
            for box, cls in zip(boxes, labels):
                x_c = (box[0] + box[2]) / 2 / target_size[0]
                y_c = (box[1] + box[3]) / 2 / target_size[1]
                bw = (box[2] - box[0]) / target_size[0]
                bh = (box[3] - box[1]) / target_size[1]

                x_c = min(max(x_c, 0.0), 1.0)
                y_c = min(max(y_c, 0.0), 1.0)
                bw = min(max(bw, 0.0), 1.0)
                bh = min(max(bh, 0.0), 1.0)

                lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

            label_path = os.path.join(lbl_out, fn.rsplit(".", 1)[0] + ".txt")
            with open(label_path, "w") as f:
                f.write("\n".join(lines))

            # Augmentation
            if augment and split == "train":
                aug_img, aug_boxes = apply_augmentation(img, boxes)

                # Resize after augmentation
                aug_img = aug_img.resize(target_size, Image.BILINEAR)

                # Rescale augmented boxes
                aug_boxes_scaled = [
                    [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
                    for b in aug_boxes
                ]

                aug_fn = fn.rsplit(".", 1)[0] + "_aug." + fn.rsplit(".", 1)[1]
                aug_img.save(os.path.join(img_out, aug_fn))

                aug_lines = []
                for box, cls in zip(aug_boxes_scaled, labels):
                    x_c = (box[0] + box[2]) / 2 / target_size[0]
                    y_c = (box[1] + box[3]) / 2 / target_size[1]
                    bw = (box[2] - box[0]) / target_size[0]
                    bh = (box[3] - box[1]) / target_size[1]

                    x_c = min(max(x_c, 0.0), 1.0)
                    y_c = min(max(y_c, 0.0), 1.0)
                    bw = min(max(bw, 0.0), 1.0)
                    bh = min(max(bh, 0.0), 1.0)

                    aug_lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

                aug_label_path = os.path.join(
                    lbl_out, aug_fn.rsplit(".", 1)[0] + ".txt"
                )
                with open(aug_label_path, "w") as f:
                    f.write("\n".join(aug_lines))

    # Save YAML
    yaml_content = {
        "path": base_out_dir,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(all_parts),
        "names": all_parts,
        "channels": 3,
    }

    with open(os.path.join(base_out_dir, "data.yaml"), "w") as yf:
        _yaml.dump(yaml_content, yf)

    print(f"YOLO dataset created at {base_out_dir} (augment={augment})")


if __name__ == "__main__":
    # Configuration
    annotation_json = "data/processed/final_annotations_without_occluded.json"
    image_directory = "data/images"
    test_ratio = 0.2
    valid_ratio = 0.1
    random_seed = 42
    limit = None

    splits, _ = prepare_splits(
        annotation_json, test_ratio, valid_ratio, random_seed, limit
    )

    # Output base folder
    base_folder = "data/yolo_format"

    # Without augmentation
    convert_json_to_yolo(
        annotation_json=annotation_json,
        image_dir=image_directory,
        base_out_dir=os.path.join(base_folder, "noaug"),
        splits=splits,
        augment=False,
    )

    # With augmentation
    convert_json_to_yolo(
        annotation_json=annotation_json,
        image_dir=image_directory,
        base_out_dir=os.path.join(base_folder, "aug"),
        splits=splits,
        augment=True,
    )
