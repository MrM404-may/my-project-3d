"""
基于距离的高斯球颜色赋值系统

功能：
1. 记录高斯球在不同相机视角下的属性（位置、颜色、cov、透明度）
2. 根据高斯球到相机的距离，将颜色重新赋值为定值

使用方法：
1. 在渲染时调用 record_gaussian_properties() 记录属性
2. 在需要时调用 apply_distance_based_color() 应用基于距离的颜色
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class GaussianPropertyRecorder:
    """
    记录高斯球在不同相机下的属性
    
    数据结构：
    - gaussian_id: 高斯球的全局索引
    - position_3d: 3D位置
    - camera_center: 相机中心位置
    - distance: 高斯球到相机的距离
    - color: RGB颜色
    - cov: 协方差矩阵（可选）
    - opacity: 透明度
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.reset()
    
    def reset(self):
        """清空所有记录"""
        self.records = {
            'gaussian_ids': [],
            'positions': [],
            'camera_centers': [],
            'distances': [],
            'colors': [],
            'covs': [],
            'opacities': []
        }
        self.record_count = 0
    
    def record(self, 
               gaussian_ids: torch.Tensor,
               positions: torch.Tensor,
               colors: torch.Tensor,
               covs: Optional[torch.Tensor] = None,
               opacities: torch.Tensor = None,
               camera_center: torch.Tensor = None):
        """
        记录单个相机的渲染结果
        
        参数：
        - gaussian_ids: 高斯球的索引 (N,)
        - positions: 高斯球位置 (N, 3)
        - colors: 颜色值 (N, 3)
        - covs: 协方差 (N, 6) 或 None
        - opacities: 透明度 (N, 1) 或 (N,)
        - camera_center: 相机中心位置 (3,) 用于计算距离
        """
        # 计算到相机的距离
        if camera_center is not None:
            distances = torch.norm(positions - camera_center.unsqueeze(0), dim=1)
        else:
            distances = torch.zeros(positions.shape[0], device=self.device)
        
        # 存储数据
        self.records['gaussian_ids'].append(gaussian_ids.cpu())
        self.records['positions'].append(positions.cpu())
        self.records['camera_centers'].append(camera_center.cpu() if camera_center is not None else None)
        self.records['distances'].append(distances.cpu())
        self.records['colors'].append(colors.cpu())
        self.records['covs'].append(covs.cpu() if covs is not None else None)
        self.records['opacities'].append(opacities.cpu() if opacities is not None else None)
        
        self.record_count += 1
    
    def get_unique_gaussians(self) -> torch.Tensor:
        """获取所有被记录的高斯球ID（去重）"""
        all_ids = torch.cat(self.records['gaussian_ids'])
        return torch.unique(all_ids)
    
    def get_gaussian_properties(self, gaussian_id: int) -> Dict:
        """
        获取指定高斯球在所有相机下的属性
        
        返回：
        - distances: 到各相机的距离列表
        - colors: 各相机下的颜色列表
        - average_color: 平均颜色
        - min_distance_color: 最近距离时的颜色
        """
        distances = []
        colors = []
        
        for i in range(self.record_count):
            ids = self.records['gaussian_ids'][i]
            mask = ids == gaussian_id
            
            if torch.any(mask):
                distances.append(self.records['distances'][i][mask])
                colors.append(self.records['colors'][i][mask])
        
        if not distances:
            return None
        
        distances = torch.cat(distances)
        colors = torch.cat(colors)
        
        # 按距离排序
        sorted_indices = torch.argsort(distances)
        
        return {
            'distances': distances,
            'colors': colors,
            'sorted_distances': distances[sorted_indices],
            'sorted_colors': colors[sorted_indices],
            'average_color': colors.mean(dim=0),
            'min_distance': distances.min(),
            'max_distance': distances.max(),
        }
    
    def get_distance_color_mapping(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取所有高斯球的距离-颜色映射
        
        返回：
        - all_distances: 所有距离值
        - all_colors: 对应的颜色值
        """
        all_distances = torch.cat(self.records['distances'])
        all_colors = torch.cat(self.records['colors'])
        
        return all_distances, all_colors


class DistanceBasedColorAssigner:
    """
    基于距离的颜色赋值器
    
    支持的颜色赋值策略：
    1. constant: 所有高斯球使用同一个颜色
    2. distance_gradient: 根据距离渐变
    3. nearest_color: 使用最近距离时的颜色
    4. weighted_average: 根据距离加权平均
    """
    
    def __init__(self, recording: GaussianPropertyRecorder, device='cuda'):
        self.recording = recording
        self.device = device
        self.color_mode = 'constant'
        self.constant_color = torch.tensor([1.0, 0.0, 0.0], device=device)  # 默认红色
    
    def set_constant_color(self, color: Tuple[float, float, float]):
        """设置常量颜色"""
        self.constant_color = torch.tensor(color, dtype=torch.float32, device=self.device)
        self.color_mode = 'constant'
    
    def set_distance_gradient(self, 
                              near_color: Tuple[float, float, float],
                              far_color: Tuple[float, float, float]):
        """
        设置距离渐变颜色
        
        near_color: 近处颜色
        far_color: 远处颜色
        """
        self.near_color = torch.tensor(near_color, dtype=torch.float32, device=self.device)
        self.far_color = torch.tensor(far_color, dtype=torch.float32, device=self.device)
        self.color_mode = 'distance_gradient'
    
    def set_nearest_color_mode(self):
        """使用最近距离时的颜色"""
        self.color_mode = 'nearest_color'
    
    def set_weighted_average_mode(self):
        """使用距离加权平均"""
        self.color_mode = 'weighted_average'
    
    def assign_colors(self, 
                      gaussian_ids: torch.Tensor,
                      positions: torch.Tensor,
                      original_colors: torch.Tensor,
                      distances: Optional[torch.Tensor] = None,
                      camera_center: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        根据策略分配颜色
        
        参数：
        - gaussian_ids: 高斯球索引
        - positions: 位置
        - original_colors: 原始颜色
        - distances: 预计算的距离（可选）
        - camera_center: 相机中心（如果 distances 为 None）
        
        返回：
        - 重新分配的颜色
        """
        # 计算距离
        if distances is None and camera_center is not None:
            distances = torch.norm(positions - camera_center.unsqueeze(0), dim=1)
        elif distances is None:
            distances = torch.zeros(positions.shape[0], device=self.device)
        
        # 根据模式分配颜色
        if self.color_mode == 'constant':
            return self.constant_color.unsqueeze(0).expand_as(original_colors)
        
        elif self.color_mode == 'distance_gradient':
            # 归一化距离
            if distances.max() > 0:
                normalized_dist = distances / distances.max()
            else:
                normalized_dist = torch.zeros_like(distances)
            
            # 插值颜色
            colors = self.near_color.unsqueeze(0) * (1 - normalized_dist.unsqueeze(1)) + \
                    self.far_color.unsqueeze(0) * normalized_dist.unsqueeze(1)
            return colors
        
        elif self.color_mode == 'nearest_color':
            # 为每个高斯球找到最近距离时的颜色
            assigned_colors = []
            for i, gid in enumerate(gaussian_ids):
                props = self.recording.get_gaussian_properties(gid.item())
                if props is not None:
                    nearest_idx = torch.argmin(props['distances'])
                    assigned_colors.append(props['colors'][nearest_idx])
                else:
                    assigned_colors.append(original_colors[i])
            return torch.stack(assigned_colors)
        
        elif self.color_mode == 'weighted_average':
            # 距离加权平均
            assigned_colors = []
            for i, gid in enumerate(gaussian_ids):
                props = self.recording.get_gaussian_properties(gid.item())
                if props is not None and len(props['distances']) > 0:
                    # 距离越小权重越大
                    weights = 1.0 / (props['distances'] + 1e-6)
                    weights = weights / weights.sum()
                    avg_color = (props['colors'] * weights.unsqueeze(1)).sum(dim=0)
                    assigned_colors.append(avg_color)
                else:
                    assigned_colors.append(original_colors[i])
            return torch.stack(assigned_colors)
        
        else:
            return original_colors


class IntegratedColorModifier:
    """
    集成到渲染管线的颜色修改器
    
    使用方法：
    1. 初始化时传入 GaussianModel
    2. 在 render() 前后调用相关方法
    3. 自动处理记录和应用
    """
    
    def __init__(self, gaussian_model, device='cuda'):
        self.gaussian_model = gaussian_model
        self.device = device
        self.recorder = GaussianPropertyRecorder(device=device)
        self.assigner = DistanceBasedColorAssigner(self.recorder, device=device)
        self.enabled = False
        self.record_enabled = True
        
    def enable_recording(self):
        """启用记录"""
        self.record_enabled = True
    
    def disable_recording(self):
        """禁用记录"""
        self.record_enabled = False
    
    def pre_render_record(self, 
                         gaussian_ids: torch.Tensor,
                         positions: torch.Tensor,
                         colors: torch.Tensor,
                         covs: Optional[torch.Tensor] = None,
                         opacities: torch.Tensor = None,
                         camera_center: torch.Tensor = None):
        """
        在渲染前记录高斯球属性
        
        在 gaussian_renderer/__init__.py 的 generate_neural_gaussians 函数中调用
        """
        if self.record_enabled:
            self.recorder.record(
                gaussian_ids=gaussian_ids,
                positions=positions,
                colors=colors,
                covs=covs,
                opacities=opacities,
                camera_center=camera_center
            )
    
    def post_render_assign(self,
                          gaussian_ids: torch.Tensor,
                          positions: torch.Tensor,
                          original_colors: torch.Tensor,
                          distances: Optional[torch.Tensor] = None,
                          camera_center: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        在渲染后根据距离重新分配颜色
        
        返回重新分配的颜色
        """
        if not self.enabled:
            return original_colors
        
        return self.assigner.assign_colors(
            gaussian_ids=gaussian_ids,
            positions=positions,
            original_colors=original_colors,
            distances=distances,
            camera_center=camera_center
        )
    
    def reset_recording(self):
        """重置记录"""
        self.recorder.reset()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        unique_gaussians = self.recorder.get_unique_gaussians()
        return {
            'total_records': self.recorder.record_count,
            'unique_gaussians': len(unique_gaussians),
            'current_mode': self.assigner.color_mode,
            'recording_enabled': self.record_enabled,
            'modifier_enabled': self.enabled
        }
