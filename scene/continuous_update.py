#
# Copyright (C) 2026 - Continuous Update for Octree-GS
# Based on GaussianUpdate: Continual 3D Gaussian Splatting Update
#

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import os

# ==============================
# 全局外观模型 (4D Hash Grid + Tiny MLP)
# 输入：3D位置 + 时间戳 + 区域ID
# 输出：增量球谐系数 + 增量缩放
# ==============================
class HashEncoding(nn.Module):
    """
    4D Hash Grid 编码：支持位置(x,y,z) + 时间(t) + 区域(region)作为输入
    参考论文的实现方式，使用多层哈希表
    """
    def __init__(
        self,
        num_levels: int = 16,          # 哈希表层数
        min_res: int = 16,             # 最粗分辨率
        max_res: int = 512,            # 最细分辨率
        log2_hashmap_size: int = 21,   # 哈希表大小 2^21
        features_per_level: int = 4,  # 每层特征维度
        base_dim: int = 5,             # 基础输入维度: x,y,z,t,region
    ):
        super().__init__()
        self.num_levels = num_levels
        self.min_res = min_res
        self.max_res = max_res
        self.log2_hashmap_size = log2_hashmap_size
        self.features_per_level = features_per_level
        self.base_dim = base_dim
        
        # 计算每层的分辨率缩放系数
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1))
        
        # 创建哈希嵌入层
        self.embeddings = nn.ModuleList()
        self.scalings = []
        
        for i in range(num_levels):
            resolution = int(min_res * (growth_factor ** i))
            self.scalings.append(resolution)
            
            # 哈希表的大小，最多是 2^21
            hashmap_size = min(2 ** log2_hashmap_size, resolution ** base_dim)
            embedding = nn.Embedding(hashmap_size, features_per_level)
            # 初始化权重
            nn.init.uniform_(embedding.weight, -1e-4, 1e-4)
            self.embeddings.append(embedding)
    
    def hash_fn(self, coords: torch.Tensor, level: int) -> torch.Tensor:
        """
        简单的哈希函数，将 5D 坐标映射到哈希表索引
        coords: [N, 5] (x,y,z,t,region)
        """
        # 使用质数进行哈希
        primes = [1, 2654435761, 805459861, 3674653429, 2097192037]
        x = coords.long()
        
        # 将坐标转换为整数并缩放
        scaling = self.scalings[level]
        scaled_coords = torch.floor(coords * scaling).long()
        
        # 计算哈希值
        hash_val = torch.zeros_like(scaled_coords[:, 0])
        for i in range(self.base_dim):
            hash_val ^= scaled_coords[:, i] * primes[i]
        
        # 确保哈希值在合法范围内
        hash_val = torch.abs(hash_val) % (2 ** self.log2_hashmap_size)
        return hash_val
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, 5] (x,y,z,t,region)
        输出: [N, num_levels * features_per_level]
        """
        encoded_features = []
        
        for level in range(self.num_levels):
            # 计算哈希索引
            hash_idx = self.hash_fn(x, level)
            # 获取特征
            features = self.embeddings[level](hash_idx)
            encoded_features.append(features)
        
        # 拼接所有层的特征
        return torch.cat(encoded_features, dim=-1)


class TinyMLP(nn.Module):
    """
    小MLP用于解码哈希特征到外观属性
    输入：哈希特征 + 位置
    输出：球谐系数增量 + 缩放增量
    """
    def __init__(
        self,
        input_dim: int = 16 * 4,  # 16层 * 4特征
        hidden_dim: int = 128,
        sh_degree: int = 3,      # 球谐系数度数
    ):
        super().__init__()
        
        # 计算球谐系数数量 (3颜色通道 * (degree+1)^2)
        sh_coeffs_count = 3 * (sh_degree + 1) ** 2
        
        # 输出维度：球谐系数增量 + 3个缩放增量
        self.output_dim = sh_coeffs_count + 3
        
        # MLP结构
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GlobalAppearanceModel(nn.Module):
    """
    全局外观模型：结合Hash Encoding和Tiny MLP
    用于学习不同时间、不同区域的外观变化
    """
    def __init__(
        self,
        num_regions: int = 65,  # 最大区域数量
        sh_degree: int = 3,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.sh_degree = sh_degree
        
        self.hash_encoding = HashEncoding()
        self.mlp = TinyMLP(sh_degree=sh_degree)
        
        # 球谐系数数量
        self.sh_coeffs_count = 3 * (sh_degree + 1) ** 2
    
    def forward(
        self, 
        positions: torch.Tensor, 
        timestamp: float, 
        region_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入：
            positions: [N, 3] - 高斯球中心位置
            timestamp: float - 当前时间戳 (归一化到 [0, 1])
            region_ids: [N] - 每个高斯的区域ID
        
        输出：
            delta_sh: [N, sh_coeffs_count] - 球谐系数增量
            delta_scale: [N, 3] - 缩放增量
        """
        N = positions.shape[0]
        
        # 构建 5D 输入：[x, y, z, t, region]
        t_tensor = torch.full((N, 1), timestamp, device=positions.device, dtype=positions.dtype)
        region_tensor = region_ids.unsqueeze(1).to(dtype=positions.dtype)
        
        # 拼接输入
        x = torch.cat([positions, t_tensor, region_tensor], dim=-1)  # [N, 5]
        
        # Hash 编码
        hash_features = self.hash_encoding(x)
        
        # MLP 解码
        output = self.mlp(hash_features)
        
        # 拆分输出
        delta_sh = output[:, :self.sh_coeffs_count]
        delta_scale = output[:, self.sh_coeffs_count:]
        
        return delta_sh, delta_scale


# ==============================
# 可见性池 (Visibility Pool)
# 管理所有历史时刻的高斯球和相机
# ==============================
class VisibilityPool:
    """
    可见性池：存储历史高斯球状态和相机信息
    """
    def __init__(self):
        # 按时间戳存储高斯球状态
        self.gaussian_states = {}  # {timestamp: state_dict}
        
        # 按时间戳存储相机信息
        self.camera_states = {}  # {timestamp: camera_info}
        
        # 高斯球的活跃状态标记
        self.gaussian_activity = defaultdict(set)  # {timestamp: active_indices}
        
        # 历史相机外参 (用于生成回放)
        self.historical_cameras = []  # 列表形式存储所有历史相机
    
    def save_gaussian_state(self, timestamp: float, gaussians: Any):
        """保存当前时刻的高斯球状态"""
        state = {
            'anchors': gaussians._anchor.clone().detach().cpu(),
            'scales': gaussians._scaling.clone().detach().cpu(),
            'rotations': gaussians._rotation.clone().detach().cpu(),
            'opacities': gaussians._opacity.clone().detach().cpu(),
            'anchor_feats': gaussians._anchor_feat.clone().detach().cpu(),
            'regions': gaussians._region.clone().detach().cpu(),
            'levels': gaussians._level.clone().detach().cpu(),
        }
        self.gaussian_states[timestamp] = state
    
    def save_camera_state(self, timestamp: float, cameras: List[Any]):
        """保存当前时刻的相机信息"""
        camera_info = []
        for cam in cameras:
            info = {
                'uid': cam.uid,
                'camera_center': cam.camera_center.clone().detach().cpu(),
                'world_view_transform': cam.world_view_transform.clone().detach().cpu(),
                'full_proj_transform': cam.full_proj_transform.clone().detach().cpu(),
                'image_name': getattr(cam, 'image_name', f'cam_{cam.uid}'),
            }
            camera_info.append(info)
        self.camera_states[timestamp] = camera_info
        
        # 添加到历史相机列表
        self.historical_cameras.extend(camera_info)
    
    def mark_active_gaussians(self, timestamp: float, active_indices: torch.Tensor):
        """标记当前活跃的高斯球"""
        self.gaussian_activity[timestamp] = set(active_indices.tolist())
    
    def get_gaussian_state(self, timestamp: float) -> Optional[Dict]:
        """获取指定时刻的高斯球状态"""
        return self.gaussian_states.get(timestamp, None)
    
    def get_historical_cameras(self) -> List[Dict]:
        """获取所有历史相机信息"""
        return self.historical_cameras


# ==============================
# 布局不变掩码 (Layout-Invariant Mask)
# 使用SAM网络生成的伪接口
# ==============================
class LayoutInvariantMaskGenerator:
    """
    布局不变掩码生成器
    使用SAM网络检测变化区域，生成掩码
    【注意】SAM接口需要后续完善
    """
    def __init__(self):
        # SAM模型占位
        self.sam_model = None
        print("[LayoutInvariantMask] 初始化完成，SAM接口待完善")
    
    def generate_mask(
        self, 
        old_image: torch.Tensor, 
        new_image: torch.Tensor, 
        region_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        生成布局不变掩码
        输入：
            old_image: 用旧模型渲染的图像
            new_image: 新时刻的真实图像
            region_id: 区域ID (可选)
        输出：
            mask: [H, W] 布尔掩码，True表示布局不变区域
        """
        # ==============================
        # 【占位代码】SAM接口实现待完善
        # ==============================
        
        # 临时实现：基于图像差异的简单掩码
        # 实际应使用SAM进行实例分割，计算IoU
        diff = torch.abs(old_image - new_image).mean(dim=0)  # [H, W]
        # 差异小的区域认为布局不变
        mask = diff < 0.1  # 阈值可调整
        
        return mask
    
    def load_sam_model(self, model_path: str):
        """加载SAM模型 (后续实现)"""
        pass


# ==============================
# 移除因子 (Removal Factor)
# 用于处理消失的物体
# ==============================
class RemovalFactor(nn.Module):
    """
    可学习的移除因子
    用于标记可能消失的高斯球
    """
    def __init__(self, num_gaussians: int):
        super().__init__()
        # 初始化移除因子，初始值设为0，表示活跃
        self.removal_factors = nn.Parameter(torch.zeros(num_gaussians, 1))
    
    def get_removal_prob(self) -> torch.Tensor:
        """
        获取移除概率，使用 sigmoid 函数
        参考论文：ψ(m) = 1 / (1 + exp(-1000 * m))
        """
        return torch.sigmoid(1000 * self.removal_factors)
    
    def apply(self, opacities: torch.Tensor) -> torch.Tensor:
        """
        应用移除因子到不透明度
        """
        removal_prob = self.get_removal_prob()
        return opacities * (1 - removal_prob)
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        计算正则化损失
        鼓励移除因子趋向于0或1
        """
        prob = self.get_removal_prob()
        # L2损失，鼓励prob接近0或1
        reg_loss = torch.mean((1 - prob) * prob)
        # 额外的二值化损失
        bce_loss = torch.mean(prob * torch.log(prob + 1e-6) + (1 - prob) * torch.log(1 - prob + 1e-6))
        return -0.01 * reg_loss + 0.001 * bce_loss


# ==============================
# DBSCAN 聚类剪枝
# ==============================
class DBSCANPruner:
    """
    使用DBSCAN聚类进行剪枝
    """
    def __init__(self, eps: float = 0.1, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
    
    def fit_predict(self, points: torch.Tensor) -> torch.Tensor:
        """
        简单的DBSCAN实现 (GPU加速)
        输入：points: [N, 3]
        输出：labels: [N] 聚类标签，-1表示噪声
        """
        N = points.shape[0]
        labels = torch.full((N,), -1, device=points.device, dtype=torch.long)
        cluster_id = 0
        
        # 计算距离矩阵
        dist_matrix = torch.cdist(points, points)
        
        for i in range(N):
            if labels[i] != -1:
                continue
            
            # 找到所有邻域点
            neighbors = torch.where(dist_matrix[i] < self.eps)[0]
            
            if len(neighbors) < self.min_samples:
                # 噪声点
                labels[i] = -1
            else:
                # 扩展簇
                cluster = [i]
                labels[i] = cluster_id
                queue = list(neighbors)
                
                while queue:
                    idx = queue.pop(0)
                    if labels[idx] == -1:
                        # 标记为当前簇
                        labels[idx] = cluster_id
                        # 找到该点的邻居
                        new_neighbors = torch.where(dist_matrix[idx] < self.eps)[0]
                        if len(new_neighbors) >= self.min_samples:
                            queue.extend(new_neighbors.tolist())
                
                cluster_id += 1
        
        return labels
    
    def prune(
        self, 
        removal_probs: torch.Tensor, 
        positions: torch.Tensor, 
        threshold: float = 0.01
    ) -> torch.Tensor:
        """
        执行剪枝
        输入：
            removal_probs: [N, 1] 移除概率
            positions: [N, 3] 高斯位置
            threshold: 移除概率阈值
        输出：
            prune_mask: [N] 布尔掩码，True表示应该被剪枝
        """
        # 首先通过阈值筛选候选
        candidate_mask = removal_probs.squeeze() > threshold
        
        if not candidate_mask.any():
            return torch.zeros_like(candidate_mask, dtype=torch.bool)
        
        # 对候选点进行DBSCAN聚类
        candidate_positions = positions[candidate_mask]
        candidate_labels = self.fit_predict(candidate_positions)
        
        # 构建完整的标签向量
        labels = torch.full((positions.shape[0],), -1, device=positions.device, dtype=torch.long)
        labels[candidate_mask] = candidate_labels
        
        # 找出规模较大的簇进行剪枝
        prune_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
        
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            if label == -1:
                continue
            cluster_mask = labels == label
            cluster_size = cluster_mask.sum()
            # 剪枝足够大的簇
            if cluster_size >= self.min_samples:
                prune_mask[cluster_mask] = True
        
        return prune_mask


# ==============================
# 三阶段更新管理器
# ==============================
class ContinuousUpdateManager:
    """
    持续更新管理器
    实现三阶段更新流程：
        1. 全局外观更新
        2. 几何布局更新
        3. 联合精炼
    """
    def __init__(
        self,
        num_regions: int = 65,
        sh_degree: int = 3,
        max_timestamps: int = 100,
    ):
        self.num_regions = num_regions
        self.sh_degree = sh_degree
        self.max_timestamps = max_timestamps
        
        # 全局外观模型
        self.global_appearance = GlobalAppearanceModel(num_regions=num_regions, sh_degree=sh_degree)
        
        # 可见性池
        self.visibility_pool = VisibilityPool()
        
        # 布局不变掩码生成器
        self.mask_generator = LayoutInvariantMaskGenerator()
        
        # DBSCAN剪枝器
        self.dbscan_pruner = DBSCANPruner()
        
        # 移除因子 (在每个时间步会重新初始化)
        self.removal_factor = None
        
        # 当前时间戳
        self.current_timestamp = 0.0
        
        # 三阶段配置
        self.stage_configs = {
            'appearance': {'iterations': 7000},
            'geometry': {'iterations': 8000, 'dbscan_iters': [5000, 8000]},
            'refinement': {'iterations': 15000, 'importance_prune_iter': 4000},
        }
        
        # 跟踪当前阶段
        self.current_stage = None
        self.stage_iteration = 0
        
        print("[ContinuousUpdateManager] 初始化完成")
    
    def initialize_new_timestamp(self, timestamp: float, num_gaussians: int):
        """
        初始化新时间戳
        """
        self.current_timestamp = timestamp
        # 初始化移除因子
        self.removal_factor = RemovalFactor(num_gaussians)
        self.current_stage = 'appearance'
        self.stage_iteration = 0
        print(f"[ContinuousUpdateManager] 初始化时间戳: {timestamp}")
    
    def get_appearance_model(self) -> GlobalAppearanceModel:
        return self.global_appearance
    
    def get_removal_factor(self) -> Optional[RemovalFactor]:
        return self.removal_factor
    
    def get_mask_generator(self) -> LayoutInvariantMaskGenerator:
        return self.mask_generator
    
    def update_stage(self):
        """更新阶段状态"""
        self.stage_iteration += 1
        
        # 检查是否需要切换阶段
        if self.current_stage == 'appearance':
            if self.stage_iteration >= self.stage_configs['appearance']['iterations']:
                self.current_stage = 'geometry'
                self.stage_iteration = 0
                print("[ContinuousUpdateManager] 切换到几何布局更新阶段")
        elif self.current_stage == 'geometry':
            if self.stage_iteration >= self.stage_configs['geometry']['iterations']:
                self.current_stage = 'refinement'
                self.stage_iteration = 0
                print("[ContinuousUpdateManager] 切换到联合精炼阶段")
        elif self.current_stage == 'refinement':
            if self.stage_iteration >= self.stage_configs['refinement']['iterations']:
                print("[ContinuousUpdateManager] 更新流程完成")
    
    def should_do_dbscan(self) -> bool:
        """检查是否应该执行DBSCAN剪枝"""
        if self.current_stage != 'geometry':
            return False
        return self.stage_iteration in self.stage_configs['geometry']['dbscan_iters']
    
    def should_do_importance_prune(self) -> bool:
        """检查是否应该执行重要性剪枝"""
        if self.current_stage != 'refinement':
            return False
        return self.stage_iteration == self.stage_configs['refinement']['importance_prune_iter']
    
    def compute_importance_score(
        self,
        gaussians: Any,
        render_pkg: Dict
    ) -> torch.Tensor:
        """
        计算高斯球的重要性分数
        基于不透明度和可见性
        """
        # 这里是简化实现，实际应基于渲染中的重要性
        opacity = gaussians.get_opacity
        return opacity.squeeze()
    
    def save_current_state(self, gaussians: Any, cameras: List[Any]):
        """保存当前状态到可见性池"""
        self.visibility_pool.save_gaussian_state(self.current_timestamp, gaussians)
        self.visibility_pool.save_camera_state(self.current_timestamp, cameras)
    
    def get_generative_replay_cameras(self) -> List[Dict]:
        """获取用于生成式回放的历史相机"""
        return self.visibility_pool.get_historical_cameras()
