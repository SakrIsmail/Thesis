import os
import json
import random
import shutil
from PIL import Image
from torchvision import transforms

def set_seed(seed: int = 42):
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def prepare_splits(annotation_json, test_ratio=0.2, valid_ratio=0.1, seed=42, limit=None):
    """
    Splits the dataset filenames into train/val/test according to ratios.
    Optionally limits to first limit images.
    Returns dict: { 'train': [...], 'val': [...], 'test': [...] }
    """
    with open(annotation_json, 'r') as f:
        annotations = json.load(f)
    image_filenames = list(annotations['images'].keys())
    if limit:
        image_filenames = image_filenames[:limit]
    random.seed(seed)
    random.shuffle(image_filenames)
    n = len(image_filenames)
    n_test = int(n * test_ratio)
    n_train = n - n_test
    n_val = int(n_train * valid_ratio)
    test = image_filenames[:n_test]
    val = image_filenames[n_test:n_test + n_val]
    train = image_filenames[n_test + n_val:]
    return {
        'train': train,
        'val': val,
        'test': test
    }, annotations

def convert_json_to_yolo(annotation_json, image_dir, base_out_dir, splits, augment=False):
    with open(annotation_json, 'r') as f:
        annotations = json.load(f)
    all_parts = annotations['all_parts']
    part_to_idx = {p: i for i, p in enumerate(all_parts)}

    aug_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3)
    ]) if augment else None

    target_size = (640, 640)  # New size

    for split, filenames in splits.items():
        img_out = os.path.join(base_out_dir, 'images', split)
        lbl_out = os.path.join(base_out_dir, 'labels', split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for fn in filenames:
            src_img = os.path.join(image_dir, fn)
            img = Image.open(src_img).convert('RGB')
            orig_w, orig_h = img.size

            # Resize image
            img_resized = img.resize(target_size, Image.BILINEAR)
            img_resized.save(os.path.join(img_out, fn))

            parts = annotations['images'][fn]['available_parts']
            lines = []

            for part in parts:
                cls = part_to_idx[part['part_name']]
                bb = part['absolute_bounding_box']

                # Adjust bounding box to resized image
                scale_x = target_size[0] / orig_w
                scale_y = target_size[1] / orig_h

                left = bb['left'] * scale_x
                top = bb['top'] * scale_y
                width = bb['width'] * scale_x
                height = bb['height'] * scale_y

                x_c = (left + width / 2) / target_size[0]
                y_c = (top + height / 2) / target_size[1]
                bw = width / target_size[0]
                bh = height / target_size[1]

                # Clamp values
                x_c = min(max(x_c, 0.0), 1.0)
                y_c = min(max(y_c, 0.0), 1.0)
                bw = min(max(bw, 0.0), 1.0)
                bh = min(max(bh, 0.0), 1.0)

                lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

            label_path = os.path.join(lbl_out, fn.rsplit('.', 1)[0] + '.txt')
            with open(label_path, 'w') as f:
                f.write("\n".join(lines))

            # Augmentation (only for training set)
            if augment and split == 'train':
                aug_img = aug_transforms(img_resized)
                aug_fn = fn.rsplit('.', 1)[0] + '_aug.' + fn.rsplit('.', 1)[1]
                aug_img.save(os.path.join(img_out, aug_fn))

                aug_lines = []
                for part in parts:
                    cls = part_to_idx[part['part_name']]
                    bb = part['absolute_bounding_box']

                    # Recompute scaled values for flipped image
                    left = bb['left'] * scale_x
                    width = bb['width'] * scale_x
                    top = bb['top'] * scale_y
                    height = bb['height'] * scale_y

                    x_c = 1.0 - ((left + width / 2) / target_size[0])  # flipped horizontally
                    y_c = (top + height / 2) / target_size[1]
                    bw = width / target_size[0]
                    bh = height / target_size[1]

                    x_c = min(max(x_c, 0.0), 1.0)
                    y_c = min(max(y_c, 0.0), 1.0)
                    bw = min(max(bw, 0.0), 1.0)
                    bh = min(max(bh, 0.0), 1.0)

                    aug_lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

                aug_label_path = os.path.join(lbl_out, aug_fn.rsplit('.', 1)[0] + '.txt')
                with open(aug_label_path, 'w') as f:
                    f.write("\n".join(aug_lines))

    # Save YAML config
    nc = len(all_parts)
    yaml_content = {
        'path': base_out_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': nc,
        'names': all_parts,
        'channels': 3
    }
    import yaml as _yaml
    with open(os.path.join(base_out_dir, 'data.yaml'), 'w') as yf:
        _yaml.dump(yaml_content, yf)

    print(f"YOLO dataset created at {base_out_dir} (augment={augment})")


if __name__ == '__main__':
    # Configuration
    annotation_json = 'data/processed/final_annotations_without_occluded.json'
    image_directory = 'data/images'
    test_ratio = 0.2
    valid_ratio = 0.1
    random_seed = 42
    limit = None

    splits, _ = prepare_splits(annotation_json, test_ratio, valid_ratio, random_seed, limit)

    # Output base folder
    base_folder = 'data/yolo_format'

    # Without augmentation
    convert_json_to_yolo(
        annotation_json=annotation_json,
        image_dir=image_directory,
        base_out_dir=os.path.join(base_folder, 'noaug'),
        splits=splits,
        augment=False
    )

    # With augmentation
    convert_json_to_yolo(
        annotation_json=annotation_json,
        image_dir=image_directory,
        base_out_dir=os.path.join(base_folder, 'aug'),
        splits=splits,
        augment=True
    )