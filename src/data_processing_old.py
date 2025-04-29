import json

def create_new_annotations(json_file, output_json_file):
    
    excluded_images = {
        "50191.jpg", "50825.jpg", "51452.jpg", "51554.jpg", "F0150f-2015000510.jpg",
        "F0150f-2016000123.jpg", "F0150f-2016000657.jpg", "F0153f-2016000414.jpg",
        "F0193f-193305753.jpg", "F0193f-193307592.jpg", "F0193f-193701242.jpg",
        "F0193f-193702806.jpg", "F0193f-193703007.jpg", "F0200a-2016000689.jpg",
        "F0268a-2016001698.jpg", "F0344a-10053114.jpg", "F0344a-10063943.jpg",
        "F0758f-758400489.jpg", "F0772f-2016000261.jpg", "F0772f-2016000806.jpg",
        "F0796f-2016000357.jpg", "G0267-2014000203.jpg", "G0537-2015001105.jpg", "G0537-2016000047.jpg"
    }

    with open(json_file, 'r') as f:
        annotations = json.load(f)
    
    image_annotations = {}
    all_parts = set()

    for image_name, image_data in annotations.items():
        if image_name in excluded_images:
            continue

        image_annotations[image_name] = {
            "available_parts": [],
            "missing_parts": []
        }
        
        for part_name, part_data in image_data["parts"].items():
            all_parts.add(part_name)

            object_state = part_data["object_state"]
            absolute_bbox = part_data.get("absolute_bounding_box")
            
            if absolute_bbox:
                absolute_bbox = {
                    "left": absolute_bbox["left"],
                    "top": absolute_bbox["top"],
                    "width": absolute_bbox["width"],
                    "height": absolute_bbox["height"]
                }

            if object_state in ["absent", "occluded"]:
                image_annotations[image_name]["missing_parts"].append(part_name)
            else:
                image_annotations[image_name]["available_parts"].append({
                    "part_name": part_name,
                    "absolute_bounding_box": absolute_bbox
                })

    annotations_with_parts = {
        "all_parts": list(all_parts),
        "images": image_annotations
    }

    with open(output_json_file, 'w') as output_json:
        json.dump(annotations_with_parts, output_json, indent=4)

    print(f"New annotations saved to {output_json_file}")

json_file = 'DelftBikes/train_annotations.json'
output_json_file = 'DelftBikes/new_annotations.json'

if __name__ == "__main__":
    create_new_annotations(json_file, output_json_file)
