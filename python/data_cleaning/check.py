import json

# checkpoint_file = 'DelftBikes/checkpoint.json'
annotations_file = 'data/processed/new_annotations.json'
final_annotation_file = 'data/processed/final_annotations.json'

MIN_PARTS = 22


all_parts = [
    "back_wheel", "back_handle", "bell", "chain", "steer", "lock", "back_light", 
    "front_handle", "dress_guard", "front_pedal", "back_pedal", "front_handbreak", 
    "saddle", "kickstand", "back_reflector", "front_wheel", "dynamo", "front_light", 
    "back_hand_break", "gear_case", "back_mudguard", "front_mudguard"
]
# def check_annotations():
#     with open(annotations_file, 'r') as f:
#         annotations_data = json.load(f)

#     print("Checking train_annotations.json for missing parts:\n")

#     for image_name, entry in annotations_data.items():
#         parts = entry.get("parts", {})
#         part_names = set(parts.keys())
#         missing_parts = [p for p in all_parts if p not in part_names]
        
#         if missing_parts:
#             print(f"{image_name} is missing {len(missing_parts)} parts:")
#             for mp in missing_parts:
#                 print(f"  - {mp}")
#             print()


# def check_checkpoint():
#     with open(checkpoint_file, 'r') as f:
#         checkpoint_data = json.load(f)

#     print(f"Images in checkpoint.json with fewer than {MIN_PARTS} parts:\n")

#     for image_name, parts in checkpoint_data.items():
#         part_count = len(parts)
#         if part_count < MIN_PARTS:
#             print(f"{image_name}: {part_count} parts")

def compare_annotations():
    with open(annotations_file, 'r') as f1, open(final_annotation_file, 'r') as f2:
        annotations_data = json.load(f1)["images"]
        cleaned_data = json.load(f2)["images"]

    for image_name in annotations_data:
        if image_name not in cleaned_data:
            print(f"{image_name} is missing completely from cleaned_annotation.json")
            continue

        parts_ann = set(part["part_name"] for part in annotations_data[image_name]["available_parts"])
        parts_clean = set(part["part_name"] for part in cleaned_data[image_name]["available_parts"])

        missing_in_cleaned = parts_ann - parts_clean
        extra_in_cleaned = parts_clean - parts_ann

        if missing_in_cleaned or extra_in_cleaned:
            print(f"{image_name} has part differences:")
            if missing_in_cleaned:
                print(f"  Missing in cleaned: {sorted(missing_in_cleaned)}")
            if extra_in_cleaned:
                print(f"  Extra in cleaned: {sorted(extra_in_cleaned)}")
            print()


# check_checkpoint()
compare_annotations()
