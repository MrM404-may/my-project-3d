#!/usr/bin/env python3
"""
训练后应用距离基础颜色的示例脚本

使用方法：
1. 训练完成后
2. 运行此脚本
3. 将处理后的模型用于渲染
"""

import sys
import os
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene.gaussian_model import GaussianModel
from post_training_color_processor import PostTrainingColorProcessor, SimpleColorAssigner


def apply_distance_color_after_training():
    """
    训练后应用距离基础颜色的主流程
    """
    print("=" * 60)
    print("Post-Training Distance-Based Color Assignment")
    print("=" * 60)
    
    # 1. 加载训练好的模型
    model_path = "/root/autodl-tmp/Octree-GS/Octree-GS/output/Ma0421"
    checkpoint_path = os.path.join(model_path, "point_cloud", "iteration_7000")
    
    print("\n[1/5] Loading trained model...")
    gaussians = GaussianModel(...)  # 根据你的初始化参数
    gaussians.load_ply(os.path.join(checkpoint_path, "point_cloud.ply"))
    
    # 2. 创建处理器
    print("\n[2/5] Creating color processor...")
    processor = PostTrainingColorProcessor(gaussians, device='cuda')
    assigner = SimpleColorAssigner(gaussians, device='cuda')
    
    # 3. 设置颜色模式
    print("\n[3/5] Selecting color mode...")
    print("Available modes:")
    print("  1. constant    - All gaussians same color")
    print("  2. gradient    - Distance-based gradient")
    print("  3. depth_encoding - Grayscale depth")
    
    color_mode = 'gradient'  # 可选：constant, gradient, depth_encoding
    
    # 4. 应用颜色
    print(f"\n[4/5] Applying {color_mode} color...")
    
    if color_mode == 'constant':
        # 示例：全部设为红色
        assigner.modify_model_color(
            positions=gaussians._anchor,
            camera_center=torch.tensor([0, 0, 0], device='cuda'),
            color_mode='constant',
            constant_color=(1.0, 0.0, 0.0)  # 红色
        )
    
    elif color_mode == 'gradient':
        # 示例：近处绿色，远处蓝色
        assigner.modify_model_color(
            positions=gaussians._anchor,
            camera_center=torch.tensor([0, 0, 0], device='cuda'),
            color_mode='gradient',
            near_color=(0.0, 1.0, 0.0),  # 绿色
            far_color=(0.0, 0.0, 1.0)     # 蓝色
        )
    
    elif color_mode == 'depth_encoding':
        # 示例：深度灰度编码
        assigner.modify_model_color(
            positions=gaussians._anchor,
            camera_center=torch.tensor([0, 0, 0], device='cuda'),
            color_mode='depth_encoding'
        )
    
    # 5. 保存处理后的模型
    print("\n[5/5] Saving processed model...")
    output_path = os.path.join(model_path, "point_cloud_colored", "iteration_7000")
    os.makedirs(output_path, exist_ok=True)
    
    processor.save_processed_model(os.path.join(output_path, "model.pt"))
    
    print("\n" + "=" * 60)
    print("Done! Processed model saved.")
    print("=" * 60)
    
    return output_path


def apply_color_during_render():
    """
    在渲染时应用颜色（不需要修改模型）
    """
    from gaussian_renderer import render
    from scene.cameras import Camera
    
    # 加载模型
    gaussians = GaussianModel(...)
    gaussians.load_ply("path/to/point_cloud.ply")
    
    # 创建赋值器
    assigner = SimpleColorAssigner(gaussians, device='cuda')
    
    # 创建相机
    camera = Camera(...)
    
    # 渲染前计算颜色
    positions = gaussians._anchor
    
    new_colors = assigner.assign_by_distance(
        positions=positions,
        camera_center=camera.camera_center,
        color_mode='gradient',
        near_color=(0.0, 1.0, 0.0),
        far_color=(0.0, 0.0, 1.0)
    )
    
    # 在渲染时使用新颜色
    # （需要修改 render 函数或使用 override_color）
    rendered_image = render(
        camera, 
        gaussians, 
        pipe, 
        bg_color,
        override_color=new_colors  # 如果 render 支持的话
    )
    
    return rendered_image


if __name__ == "__main__":
    print("Post-Training Color Processor")
    print("\nThis script shows how to apply distance-based colors")
    print("after training is complete.\n")
    
    # 示例用法
    # apply_distance_color_after_training()
    
    print("Please modify the paths and parameters as needed.")
    print("See post_training_color_processor.py for API details.")
