#!/usr/bin/env python3
"""
将旧的 anchor_features.pt 转换为 PLY 格式
"""
import os
import sys
import torch


def convert_pt_to_ply(pt_path, ply_base_path=None):
    """转换 .pt 到 PLY 格式"""
    if ply_base_path is None:
        ply_base_path = os.path.splitext(pt_path)[0]
    
    print(f"Converting {pt_path} -> {ply_base_path}")
    
    # 加载旧数据
    print("Loading data...")
    storage_dict = torch.load(pt_path)
    print(f"Loaded {len(storage_dict)} entries")
    
    # 分组
    print("Grouping data...")
    from collections import defaultdict
    groups = defaultdict(lambda: {'positions': [], 'features': []})
    
    for key, feat in storage_dict.items():
        region, moment, pos = key
        groups[(region, moment)]['positions'].append(pos)
        groups[(region, moment)]['features'].append(feat)
    
    print(f"Found {len(groups)} groups")
    
    # 保存为 PLY
    print("Saving as PLY...")
    from anchor_feature_storage_ply import AnchorFeatureStoragePLY
    storage = AnchorFeatureStoragePLY(ply_base_path, lazy_load=False)
    
    for (region, moment), data in groups.items():
        positions = data['positions']
        features = data['features']
        print(f"  Saving region={region}, moment={moment}: {len(positions)} entries")
        storage.add(region, moment, positions, features)
    
    print("Done!")
    return ply_base_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <anchor_features.pt> [output_base_path]")
        sys.exit(1)
    
    pt_path = sys.argv[1]
    ply_base_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_pt_to_ply(pt_path, ply_base_path)
