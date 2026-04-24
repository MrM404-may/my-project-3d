import os
import numpy as np
from plyfile import PlyData, PlyElement

def merge_region_anchors(model_path):
    """
    拼接所有区域的锚点文件，生成最终的点云文件
    参数:
        model_path: 模型保存路径
    """
    
    # 1. 收集所有区域的锚点文件
    region_anchors_dir = os.path.join(model_path, "region_anchors")
    ply_files = []
    
    if not os.path.exists(region_anchors_dir):
        print(f"❌ 区域锚点目录不存在: {region_anchors_dir}")
        return
    
    for region_dir in os.listdir(region_anchors_dir):
        if region_dir.startswith("region_"):
            ply_file = os.path.join(region_anchors_dir, region_dir, "anchors.ply")
            if os.path.exists(ply_file):
                ply_files.append(ply_file)
    
    if not ply_files:
        print("❌ 没有找到锚点文件")
        return
    
    print(f"✅ 找到 {len(ply_files)} 个区域锚点文件")
    
    # 2. 读取并拼接所有锚点文件
    all_data = []
    
    for ply_file in ply_files:
        print(f"📖 读取锚点文件: {ply_file}")
        ply_data = PlyData.read(ply_file)
        vertex_data = ply_data['vertex'].data
        all_data.append(vertex_data)
    
    # 拼接所有数据
    merged_data = np.concatenate(all_data, axis=0)
    
    print(f"✅ 拼接完成，总锚点数: {len(merged_data)}")
    
    # 3. 保存拼接后的点云文件
    output_ply = os.path.join(model_path, "merged_anchors.ply")
    print(f"💾 保存拼接后的点云文件: {output_ply}")
    
    # 创建PLY元素并保存（保留原始数据结构）
    el = PlyElement.describe(merged_data, 'vertex')
    PlyData([el]).write(output_ply)
    
    print(f"✅ 点云文件保存完成: {output_ply}")
    return output_ply

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="合并区域锚点文件")
    parser.add_argument("--model_path", type=str, required=True, help="模型保存路径")
    args = parser.parse_args()
    
    print("=" * 60)
    print("合并区域锚点文件")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print()
    
    merge_region_anchors(args.model_path)
