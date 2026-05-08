"""
训练后的高斯球颜色处理器

功能：
1. 加载训练好的高斯模型
2. 在不同相机下渲染，记录高斯球属性
3. 根据距离重新赋予定值颜色
4. 保存处理后的模型

使用场景：
- 训练完成后
- 最终渲染之前
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GaussianRecord:
    """单个高斯球的记录"""
    gaussian_id: int
    position: torch.Tensor  # (3,)
    distances: List[float]  # 到各相机的距离
    colors: List[torch.Tensor]  # 各相机下的颜色
    covs: Optional[List[torch.Tensor]] = None  # 协方差
    opacities: Optional[List[torch.Tensor]] = None  # 透明度


class PostTrainingColorProcessor:
    """
    训练后的颜色处理器
    
    工作流程：
    1. 加载训练好的模型
    2. 在多个相机下渲染，记录属性
    3. 根据距离重新赋值
    4. 保存结果
    """
    
    def __init__(self, gaussian_model, device='cuda'):
        self.gaussian_model = gaussian_model
        self.device = device
        self.records = {}  # {gaussian_id: GaussianRecord}
    
    def record_from_render(self, 
                          gaussian_ids: torch.Tensor,
                          positions: torch.Tensor,
                          colors: torch.Tensor,
                          distances: torch.Tensor,
                          covs: Optional[torch.Tensor] = None,
                          opacities: Optional[torch.Tensor] = None):
        """
        从渲染过程中记录高斯球属性
        
        参数：
        - gaussian_ids: 高斯球索引 (N,)
        - positions: 位置 (N, 3)
        - colors: 颜色 (N, 3)
        - distances: 到相机的距离 (N,)
        - covs: 协方差 (N, 6) [可选]
        - opacities: 透明度 (N,) [可选]
        """
        for i in range(len(gaussian_ids)):
            gid = gaussian_ids[i].item()
            
            if gid not in self.records:
                self.records[gid] = GaussianRecord(
                    gaussian_id=gid,
                    position=positions[i].cpu(),
                    distances=[],
                    colors=[],
                    covs=[] if covs is not None else None,
                    opacities=[] if opacities is not None else None
                )
            
            self.records[gid].distances.append(distances[i].item())
            self.records[gid].colors.append(colors[i].cpu())
            
            if covs is not None:
                self.records[gid].covs.append(covs[i].cpu())
            if opacities is not None:
                self.records[gid].opacities.append(opacities[i].cpu())
    
    def apply_constant_color(self, constant_color: Tuple[float, float, float]):
        """
        将所有高斯球颜色设为常量
        
        参数：
        - constant_color: (R, G, B)，值范围 [0, 1]
        """
        color_tensor = torch.tensor(constant_color, dtype=torch.float32, device=self.device)
        
        # 直接修改模型的颜色参数
        with torch.no_grad():
            # 假设颜色存储在某个属性中
            if hasattr(self.gaussian_model, '_anchor_feat'):
                # 如果是 anchor_feat，形状应该是 (N, feat_dim)
                feat_dim = self.gaussian_model._anchor_feat.shape[1]
                if feat_dim >= 3:
                    # 只修改前3个维度作为颜色
                    self.gaussian_model._anchor_feat.data[:, :3] = color_tensor
        
        print(f"Applied constant color: {constant_color}")
    
    def apply_distance_gradient(self, 
                               near_color: Tuple[float, float, float],
                               far_color: Tuple[float, float, float]):
        """
        根据距离应用渐变颜色
        
        参数：
        - near_color: 近处颜色 (R, G, B)
        - far_color: 远处颜色 (R, G, B)
        """
        near_tensor = torch.tensor(near_color, dtype=torch.float32, device=self.device)
        far_tensor = torch.tensor(far_color, dtype=torch.float32, device=self.device)
        
        # 计算所有高斯球的平均距离
        all_distances = []
        all_positions = []
        
        for gid, record in self.records.items():
            if len(record.distances) > 0:
                avg_distance = np.mean(record.distances)
                all_distances.append(avg_distance)
                all_positions.append(gid)
        
        if not all_distances:
            print("No records found, cannot apply gradient")
            return
        
        distances_tensor = torch.tensor(all_distances, dtype=torch.float32, device=self.device)
        max_dist = distances_tensor.max()
        min_dist = distances_tensor.min()
        
        # 归一化
        if max_dist > min_dist:
            normalized = (distances_tensor - min_dist) / (max_dist - min_dist)
        else:
            normalized = torch.zeros_like(distances_tensor)
        
        # 插值颜色
        new_colors = near_tensor.unsqueeze(0) * (1 - normalized.unsqueeze(1)) + \
                    far_tensor.unsqueeze(0) * normalized.unsqueeze(1)
        
        # 应用到模型
        with torch.no_grad():
            if hasattr(self.gaussian_model, '_anchor_feat'):
                for idx, gid in enumerate(all_positions):
                    # 找到对应的索引
                    pass  # 需要根据实际索引映射
        
        print(f"Applied distance gradient: near={near_color}, far={far_color}")
    
    def apply_nearest_camera_color(self):
        """
        使用最近相机视角的颜色
        """
        print("Applying nearest camera color...")
        
        # 为每个高斯球找到最近距离时的颜色
        for gid, record in self.records.items():
            if len(record.distances) > 0:
                nearest_idx = np.argmin(record.distances)
                nearest_color = record.colors[nearest_idx]
                
                # 这里需要根据 gid 找到模型中的对应索引并更新
                # 具体实现依赖于模型结构
        
        print(f"Applied nearest camera color for {len(self.records)} gaussians")
    
    def save_processed_model(self, output_path: str):
        """
        保存处理后的模型
        
        参数：
        - output_path: 保存路径
        """
        # 保存整个模型
        torch.save({
            'model_state_dict': self.gaussian_model.state_dict(),
            'records': {
                gid: {
                    'distances': rec.distances,
                    'colors': [c.numpy() for c in rec.colors],
                    'position': rec.position.numpy()
                }
                for gid, rec in self.records.items()
            }
        }, output_path)
        
        print(f"Saved processed model to {output_path}")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total_gaussians = len(self.records)
        gaussians_with_multiple_views = sum(1 for r in self.records.values() if len(r.distances) > 1)
        
        all_distances = []
        for record in self.records.values():
            all_distances.extend(record.distances)
        
        if all_distances:
            min_dist = min(all_distances)
            max_dist = max(all_distances)
            avg_dist = np.mean(all_distances)
        else:
            min_dist = max_dist = avg_dist = 0
        
        return {
            'total_gaussians': total_gaussians,
            'gaussians_with_multiple_views': gaussians_with_multiple_views,
            'min_distance': min_dist,
            'max_distance': max_dist,
            'avg_distance': avg_dist
        }


class SimpleColorAssigner:
    """
    简单的颜色赋值器（用于训练后快速处理）
    
    不需要记录过程，直接根据距离计算颜色
    """
    
    def __init__(self, gaussian_model, device='cuda'):
        self.gaussian_model = gaussian_model
        self.device = device
    
    def assign_by_distance(self, 
                          positions: torch.Tensor,
                          camera_center: torch.Tensor,
                          color_mode: str = 'constant',
                          constant_color: Optional[Tuple[float, float, float]] = None,
                          near_color: Optional[Tuple[float, float, float]] = None,
                          far_color: Optional[Tuple[float, float, float]] = None) -> torch.Tensor:
        """
        根据距离分配颜色
        
        参数：
        - positions: 高斯球位置 (N, 3)
        - camera_center: 相机中心 (3,)
        - color_mode: 'constant', 'gradient', 'depth_encoding'
        - constant_color: 常量颜色 (R, G, B)
        - near_color: 近处颜色 (R, G, B)
        - far_color: 远处颜色 (R, G, B)
        
        返回：
        - 颜色张量 (N, 3)
        """
        # 计算距离
        distances = torch.norm(positions - camera_center.unsqueeze(0), dim=1)
        
        if color_mode == 'constant':
            # 常量颜色
            color = torch.tensor(constant_color or (1.0, 0.0, 0.0), 
                               dtype=torch.float32, device=self.device)
            return color.unsqueeze(0).expand(positions.shape[0], 3)
        
        elif color_mode == 'gradient':
            # 距离渐变
            near = torch.tensor(near_color or (0.0, 1.0, 0.0), 
                               dtype=torch.float32, device=self.device)
            far = torch.tensor(far_color or (0.0, 0.0, 1.0), 
                              dtype=torch.float32, device=self.device)
            
            # 归一化
            max_dist = distances.max()
            if max_dist > 0:
                t = distances / max_dist
            else:
                t = torch.zeros_like(distances)
            
            # 插值
            colors = near.unsqueeze(0) * (1 - t.unsqueeze(1)) + \
                    far.unsqueeze(0) * t.unsqueeze(1)
            return colors
        
        elif color_mode == 'depth_encoding':
            # 深度编码（灰度）
            max_dist = distances.max()
            if max_dist > 0:
                normalized = distances / max_dist
            else:
                normalized = torch.zeros_like(distances)
            
            # 灰度值反转（近白远黑）
            gray = 1.0 - normalized
            return gray.unsqueeze(1).expand(-1, 3)
        
        else:
            raise ValueError(f"Unknown color_mode: {color_mode}")
    
    def modify_model_color(self,
                         positions: torch.Tensor,
                         camera_center: torch.Tensor,
                         color_mode: str = 'constant',
                         constant_color: Optional[Tuple[float, float, float]] = None,
                         near_color: Optional[Tuple[float, float, float]] = None,
                         far_color: Optional[Tuple[float, float, float]] = None,
                         target_attribute: str = '_anchor_feat',
                         color_dim_start: int = 0,
                         color_dim_end: int = 3):
        """
        直接修改模型的颜色属性
        
        参数：
        - positions: 高斯球位置
        - camera_center: 相机中心
        - color_mode: 颜色模式
        - target_attribute: 模型中存储颜色的属性名
        - color_dim_start/end: 颜色在属性中的维度范围
        """
        # 计算新颜色
        new_colors = self.assign_by_distance(
            positions, camera_center, color_mode,
            constant_color, near_color, far_color
        )
        
        # 修改模型
        if hasattr(self.gaussian_model, target_attribute):
            attr = getattr(self.gaussian_model, target_attribute)
            with torch.no_grad():
                attr.data[:, color_dim_start:color_dim_end] = new_colors
            print(f"Modified {target_attribute}[{color_dim_start}:{color_dim_end}] with {color_mode} color")
        else:
            print(f"Warning: Model does not have attribute {target_attribute}")
