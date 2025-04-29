import json
import os
import math
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


json_file = 'DelftBikes/train_annotations.json'
image_directory = 'DelftBikes/train'
output_json_file = 'DelftBikes/cleaned_annotations.json'
final_output_json='DelftBikes/final_annotations.json'

with open(json_file, 'r') as f:
    annotations = json.load(f)


excluded_images = {
    "50191.jpg", "50825.jpg", "51452.jpg", "51554.jpg", "F0150f-2015000510.jpg",
    "F0150f-2016000123.jpg", "F0150f-2016000657.jpg", "F0153f-2016000414.jpg",
    "F0193f-193305753.jpg", "F0193f-193307592.jpg", "F0193f-193701242.jpg",
    "F0193f-193702806.jpg", "F0193f-193703007.jpg", "F0200a-2016000689.jpg",
    "F0268a-2016001698.jpg", "F0344a-10053114.jpg", "F0344a-10063943.jpg",
    "F0758f-758400489.jpg", "F0772f-2016000261.jpg", "F0772f-2016000806.jpg",
    "F0796f-2016000357.jpg", "G0267-2014000203.jpg", "G0537-2015001105.jpg", "G0537-2016000047.jpg"
}

def refine_with_grabcut(image, bbox, iterations=5, shrink_limit=0.7):
    mask = np.zeros(image.shape[:2], np.uint8)
    rect = (bbox["left"], bbox["top"], bbox["width"], bbox["height"])
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        coords = cv2.findNonZero(mask2)

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            w = max(w, int(bbox["width"] * shrink_limit))
            h = max(h, int(bbox["height"] * shrink_limit))
        
            return {"left": x, "top": y, "width": w, "height": h}
        
    except Exception as e:
        print(f"GrabCut error: {e}")

    return bbox
# Resizing of the mudguards
def resize_bounding_box_to_top_right(bbox, factor=1.0):
    new_w = int(bbox["width"] * factor)
    new_h = int(bbox["height"] * factor)
    new_left = bbox["left"] + bbox["width"] - new_w
    return {"left": new_left, "top": bbox["top"], "width": new_w, "height": new_h}

# Used for fitting bounding boxes into other bounding boxes (steer and handles)
def clamp_box(x, y, w, h, box_left, box_top, box_right, box_bottom):
    x = max(box_left, min(x, box_right - w))
    y = max(box_top, min(y, box_bottom - h))
    return x, y


def adjust_boxes(image, part_boxes, visible_parts, use_grabcut):
    adjusted_boxes = {}
    steer_bbox = part_boxes.get("steer")
    front_wheel_bbox = part_boxes.get("front_wheel")
    back_handle_bbox = part_boxes.get("back_handle")
    adjusted_steer_bbox = steer_bbox.copy() if steer_bbox else None

    steer_is_wide = False

    # Checks if the steering wheel is wide compared to the front wheel
    if steer_bbox and front_wheel_bbox:
        if abs(steer_bbox["width"] - front_wheel_bbox["width"]) < 0.1 * front_wheel_bbox["width"]:
            steer_is_wide = True

    for name, bbox in part_boxes.items():
        if visible_parts and name not in visible_parts:
            continue

        adjusted_bbox = bbox.copy()

        if name in ["front_mudguard", "back_mudguard"]:
            adjusted_bbox = resize_bounding_box_to_top_right(bbox, factor=0.8)

        # Adjusts location of the steer in relation to the back_handle
        elif name == "steer" and back_handle_bbox:
            steer_center = (bbox["left"] + bbox["width"] / 2, bbox["top"] + bbox["height"] / 2)
            back_center = (back_handle_bbox["left"] + back_handle_bbox["width"] / 2,
                           back_handle_bbox["top"] + back_handle_bbox["height"] / 2)
            distance = math.dist(steer_center, back_center)
            threshold = 2 * math.hypot(back_handle_bbox["width"], back_handle_bbox["height"])

            if distance > threshold:

                if steer_is_wide:
                    offset = -30
                else:
                    offset = -5

                adjusted_bbox["left"] = int(back_center[0] - bbox["width"] / 2 + offset)
                adjusted_bbox["top"] = int(back_center[1] - bbox["height"] / 2)

            adjusted_steer_bbox = adjusted_bbox

        # Checks the position and size of the front_handle and front_handbreak compared to the steer
        elif name in ["front_handle", "front_handbreak"] and steer_is_wide and adjusted_steer_bbox and back_handle_bbox:
            target_size = {
                "width": back_handle_bbox["width"],
                "height": back_handle_bbox["height"]
            }

            sl, st = adjusted_steer_bbox["left"], adjusted_steer_bbox["top"]
            sr, sb = sl + adjusted_steer_bbox["width"], st + adjusted_steer_bbox["height"]

            if name == "front_handle":
                raw_left, raw_top = sl, st
            else:
                handle_box = adjusted_boxes.get("front_handle")
                raw_left = handle_box["left"] if handle_box else sl
                raw_top = (handle_box["top"] + handle_box["height"] + 5) if handle_box else (st + target_size["height"] + 5)

            clamped_left, clamped_top = clamp_box(raw_left, raw_top, target_size["width"], target_size["height"], sl, st, sr, sb)

            adjusted_bbox = {
                "left": clamped_left,
                "top": clamped_top,
                **target_size
            }

        # Adjusts the position of the back_hand_break to be below the back_handle
        elif name == "back_hand_break" and back_handle_bbox:
            adjusted_bbox = {
                "left": back_handle_bbox["left"],
                "top": back_handle_bbox["top"] + back_handle_bbox["height"] - 5,
                "width": back_handle_bbox["width"],
                "height": back_handle_bbox["height"]
            }

        if use_grabcut and name not in [
            "front_mudguard", "back_mudguard", "saddle",
            "front_wheel", "back_wheel", "front_pedal", "back_pedal", "steer"
        ]:
            adjusted_bbox = refine_with_grabcut(image, adjusted_bbox)

        adjusted_boxes[name] = adjusted_bbox

    return adjusted_boxes

def process_image(filename, annotations, image_directory):
    if filename in excluded_images:
        print(f"Skipping image: {filename} (excluded)")
        return None, filename

    path = os.path.join(image_directory, filename)
    image = cv2.imread(path)

    if image is None:
        print(f"Image not found or unreadable: {filename}")
        return None, filename

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    parts = annotations[filename].get("parts", {})

    part_boxes = {
        name: data["absolute_bounding_box"]
        for name, data in parts.items()
        if "absolute_bounding_box" in data
    }

    adjusted_boxes = adjust_boxes(
        image=image,
        part_boxes=part_boxes,
        visible_parts=None,
        use_grabcut=True
    )

    return adjusted_boxes, filename

def adjust_annotations(train_json_file, cleaned_json_file, output_json_file):
    with open(train_json_file, 'r') as f:
        train_data = json.load(f)

    with open(cleaned_json_file, 'r') as f:
        cleaned_data = json.load(f)

    final_annotations = {
        "all_parts": cleaned_data["all_parts"],
        "images": {}
    }

    for image_name, image_data in cleaned_data["images"].items():
        if image_name in excluded_images:
            continue

        available_parts = []
        missing_parts = []

        for part in image_data.get("available_parts", []):
            part_name = part["part_name"]
            bbox = part["absolute_bounding_box"]

            train_part_info = train_data.get(image_name, {}).get("parts", {}).get(part_name, None)
            object_state = train_part_info.get("object_state", "") if train_part_info else ""

            if object_state in ["intact", "damaged"]:
                available_parts.append({
                    "part_name": part_name,
                    "absolute_bounding_box": bbox
                })
            else:
                missing_parts.append(part_name)

        if available_parts or missing_parts:
            final_annotations["images"][image_name] = {
                "available_parts": available_parts,
                "missing_parts": missing_parts
            }

    with open(output_json_file, 'w') as f:
        json.dump(final_annotations, f, indent=4)

    print(f"Final adjusted annotations saved to: {output_json_file}")


def process_images_in_parallel(filenames, annotations, image_directory):
    updated_annotations = {}
    num_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_image, fn, annotations, image_directory): fn
            for fn in filenames
            if fn not in excluded_images
        }

        for future in tqdm(futures, desc="Processing Images", total=len(futures)):
            try:
                updated_boxes, filename = future.result()
                if updated_boxes:
                    updated_annotations[filename] = updated_boxes
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    return updated_annotations

def create_cleaned_annotations(annotations, updated_annotations):
    all_parts, image_annotations = set(), {}

    for image_name, image_data in annotations.items():
        image_annotations[image_name] = {"available_parts": [], "missing_parts": []}

        for part_name, part_data in image_data["parts"].items():
            absolute_bbox = updated_annotations.get(image_name, {}).get(part_name)
            
            if absolute_bbox:
                image_annotations[image_name]["available_parts"].append({"part_name": part_name, "absolute_bounding_box": absolute_bbox})
            else:
                image_annotations[image_name]["missing_parts"].append(part_name)
            all_parts.add(part_name)
            
    return {"all_parts": list(all_parts), "images": image_annotations}

def run_pipeline():
    filenames = sorted(annotations.keys())
    updated_annotations = process_images_in_parallel(filenames, annotations, image_directory)
    cleaned_annotations = create_cleaned_annotations(annotations, updated_annotations)

    with open(output_json_file, 'w') as output_json:
        json.dump(cleaned_annotations, output_json, indent=4)

    print(f"Processed annotations saved to {output_json_file}")

if __name__ == "__main__":
    run_pipeline()
    adjust_annotations(json_file, output_json_file, final_output_json)

