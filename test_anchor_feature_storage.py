#!/usr/bin/env python3
"""
测试 anchor feature 存储功能的示例脚本
"""
import torch
import os
from anchor_feature_storage import AnchorFeatureStorage

def test_anchor_feature_storage():
    print("=" * 60)
    print("测试 AnchorFeatureStorage 功能")
    print("=" * 60)
    
    # 1. 创建测试数据
    print("\n1. 创建测试数据...")
    num_points = 10
    feat_dim = 32
    positions = torch.randn(num_points, 3).float()  # 10个点，3维坐标
    anchor_feats = torch.randn(num_points, feat_dim).float()  # 对应的anchor features
    
    print(f"   位置张量形状: {positions.shape}")
    print(f"   特征张量形状: {anchor_feats.shape}")
    
    # 2. 初始化存储
    test_path = "test_anchor_features.pt"
    storage = AnchorFeatureStorage(test_path, format='pt')
    
    # 3. 添加第一个区域的数据
    print("\n2. 添加第一个区域的数据 (region 0, moment 0)...")
    storage.add(region=0, moment=0, positions=positions, anchor_feats=anchor_feats)
    print(f"   当前存储条目数: {len(storage)}")
    
    # 4. 添加第二个区域的数据
    print("\n3. 添加第二个区域的数据 (region 1, moment 0)...")
    positions2 = torch.randn(num_points, 3).float()
    anchor_feats2 = torch.randn(num_points, feat_dim).float()
    storage.add(region=1, moment=0, positions=positions2, anchor_feats=anchor_feats2)
    print(f"   当前存储条目数: {len(storage)}")
    
    # 5. 保存到文件
    print("\n4. 保存到文件...")
    storage.save()
    
    # 6. 重新加载验证
    print("\n5. 重新加载文件验证...")
    storage2 = AnchorFeatureStorage(test_path, format='pt')
    print(f"   加载后条目数: {len(storage2)}")
    
    # 7. 测试获取特定区域和时刻的数据
    print("\n6. 获取区域 0, moment 0 的数据...")
    pos_list, feat_list = storage2.get_region_moment(region=0, moment=0)
    print(f"   位置数量: {len(pos_list)}")
    print(f"   特征数量: {len(feat_list)}")
    
    # 8. 清理测试文件
    print("\n7. 清理测试文件...")
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_anchor_feature_storage()
