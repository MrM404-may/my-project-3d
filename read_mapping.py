import json

def get_cameras_json_id_by_mulp_id(mulp_id, mapping_path=r'/root/autodl-tmp/Octree-GS/Octree-GS/output/Ma3w-region/mulp_image_mapping.json'):
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    for item in mapping:
        if item['mulp_image_id'] == mulp_id:
            return item['image_cameras_json_id']
    return None

def get_full_mapping_by_mulp_id(mulp_id, mapping_path=r'/root/autodl-tmp/Octree-GS/Octree-GS/output/Ma3w-region/mulp_image_mapping.json'):
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    for item in mapping:
        if item['mulp_image_id'] == mulp_id:
            return item
    return None

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mulp_id = int(sys.argv[1])
    else:
        mulp_id = 400

    cameras_json_id = get_cameras_json_id_by_mulp_id(mulp_id)
    print(f"Mulp ID {mulp_id} -> Cameras.json ID: {cameras_json_id}")

    full_mapping = get_full_mapping_by_mulp_id(mulp_id)
    if full_mapping:
        print(f"\n完整映射信息:")
        print(f"  Mulp Image ID: {full_mapping['mulp_image_id']}")
        print(f"  Mulp Image Name: {full_mapping['mulp_image_name']}")
        print(f"  Mulp Position: {full_mapping['mulp_position']}")
        print(f"  Image ID: {full_mapping['image_id']}")
        print(f"  Image Name: {full_mapping['image_name']}")
        print(f"  Cameras.json ID: {full_mapping['image_cameras_json_id']}")
        print(f"  Distance: {full_mapping['distance']}")
