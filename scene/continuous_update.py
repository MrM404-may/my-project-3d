#
# Copyright (C) 2026 - 4D Hash + Tiny MLP 替代区域 MLP
# 基于 GaussianUpdate 设计
#

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class HashEncoding(nn.Module):
    """
    5D 哈希编码（4D 空间时间 + 区域）
    参考 Instant NGP 和 GaussianUpdate 的设计
    """
    def __init__(
        self,
        num_levels: int = 16,          # 哈希表层数
        min_res: int = 16,             # 最粗分辨率
        max_res: int = 512,            # 最细分辨率
        log2_hashmap_size: int = 21,   # 哈希表大小 2^21
        features_per_level: int = 4,  # 每层特征维度
        base_dim: int = 5,             # 基础输入维度: x,y,z,t,region
        include_space_time_only: bool = False  # 可选：仅空间时间，区域单独编码
    ):
        super().__init__()
        self.num_levels = num_levels
        self.min_res = min_res
        self.max_res = max_res
        self.log2_hashmap_size = log2_hashmap_size
        self.features_per_level = features_per_level
        self.base_dim = base_dim
        self.include_space_time_only = include_space_time_only
        
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
            # 初始化权重（小值初始化）
            nn.init.uniform_(embedding.weight, -1e-4, 1e-4)
            self.embeddings.append(embedding)
    
    def hash_fn(self, coords: torch.Tensor, level: int) -> torch.Tensor:
        """
        使用空间哈希函数，参考 Instant NGP
        coords: [N, 5] (x,y,z,t,region)
        """
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


class RegionEmbedding(nn.Module):
    """
    区域嵌入层（可选，用于补充区域信息）
    """
    def __init__(self, num_regions: int = 65, embedding_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_regions, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1e-4, 1e-4)
    
    def forward(self, region_ids: torch.Tensor) -> torch.Tensor:
        """
        region_ids: [N]
        输出: [N, embedding_dim]
        """
        return self.embedding(region_ids)


class TinyMLP(nn.Module):
    """
    小 MLP，用于从哈希特征解码到高斯属性
    """
    def __init__(
        self,
        input_dim: int = 16 * 4,  # 16层 * 4特征
        hidden_dim: int = 128,
        n_offsets: int = 5,
        feat_dim: int = 32,
        view_dim: int = 3,
        include_appearance: bool = True,
        appearance_dim: int = 32,
        output_opacity: bool = True,
        output_cov: bool = True,
        output_color: bool = True,
    ):
        super().__init__()
        
        self.output_opacity = output_opacity
        self.output_cov = output_cov
        self.output_color = output_color
        self.n_offsets = n_offsets
        
        # 计算额外输入维度
        extra_inputs = view_dim
        if include_appearance:
            extra_inputs += appearance_dim
        
        # 共享主干网络
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + extra_inputs, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        
        # 输出头
        if output_opacity:
            self.opacity_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, n_offsets),
                nn.Tanh()
            )
        
        if output_cov:
            self.cov_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, 7 * n_offsets),
            )
        
        if output_color:
            self.color_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, 3 * n_offsets),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        hash_features: torch.Tensor,
        view_direction: torch.Tensor,
        appearance_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        输入:
            hash_features: [N, hash_input_dim]
            view_direction: [N, view_dim]
            appearance_embedding: [N, appearance_dim] (可选)
        
        输出:
            包含 opacity, cov, color 的字典
        """
        # 拼接输入
        x = [hash_features, view_direction]
        if appearance_embedding is not None:
            x.append(appearance_embedding)
        x = torch.cat(x, dim=-1)
        
        # 主干网络
        features = self.backbone(x)
        
        # 输出头
        outputs = {}
        if self.output_opacity:
            outputs['opacity'] = self.opacity_head(features)
        if self.output_cov:
            outputs['cov'] = self.cov_head(features)
        if self.output_color:
            outputs['color'] = self.color_head(features)
        
        return outputs


class GlobalGaussianMLP(nn.Module):
    """
    全局高斯 MLP，替换所有区域 MLP
    输入：
        - 位置 + 时间戳 + 区域 (哈希编码)
        - 视角方向
        - 外观嵌入
    输出：
        - 不透明度
        - 协方差参数
        - 颜色
    """
    def __init__(
        self,
        num_regions: int = 65,
        n_offsets: int = 5,
        feat_dim: int = 32,
        view_dim: int = 3,
        appearance_dim: int = 32,
        include_separate_region_embedding: bool = True,  # 是否包含独立的区域嵌入
    ):
        super().__init__()
        
        self.n_offsets = n_offsets
        self.include_separate_region_embedding = include_separate_region_embedding
        
        # 哈希编码
        self.hash_encoding = HashEncoding()
        
        # 可选的区域嵌入
        self.region_embedding = None
        hash_input_dim = 16 * 4  # 默认16层 * 4特征
        if include_separate_region_embedding:
            self.region_embedding = RegionEmbedding(num_regions, 16)  # 16维区域嵌入
            hash_input_dim += 16  # 加上区域嵌入维度
        
        # Tiny MLP
        self.tiny_mlp = TinyMLP(
            input_dim=hash_input_dim,
            n_offsets=n_offsets,
            feat_dim=feat_dim,
            view_dim=view_dim,
            appearance_dim=appearance_dim,
            output_opacity=True,
            output_cov=True,
            output_color=True,
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        timestamp: float,
        region_ids: torch.Tensor,
        view_direction: torch.Tensor,
        appearance_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        输入:
            positions: [N, 3]
            timestamp: float (归一化到[0, 1])
            region_ids: [N]
            view_direction: [N, 3]
            appearance_embedding: [N, appearance_dim] (可选)
        
        输出:
            opacity: [N, n_offsets]
            cov: [N, 7 * n_offsets]
            color: [N, 3 * n_offsets]
        """
        N = positions.shape[0]
        
        # 构建 5D 输入：[x, y, z, t, region]
        t_tensor = torch.full((N, 1), timestamp, device=positions.device, dtype=positions.dtype)
        region_tensor = region_ids.unsqueeze(1).to(dtype=positions.dtype)
        hash_input = torch.cat([positions, t_tensor, region_tensor], dim=-1)  # [N, 5]
        
        # 哈希编码
        hash_features = self.hash_encoding(hash_input)
        
        # 可选：添加独立的区域嵌入
        if self.region_embedding is not None:
            region_emb = self.region_embedding(region_ids)
            hash_features = torch.cat([hash_features, region_emb], dim=-1)
        
        # Tiny MLP 解码
        outputs = self.tiny_mlp(
            hash_features,
            view_direction,
            appearance_embedding
        )
        
        return outputs
    
    def get_optimizable_params(self):
        """
        获取可优化参数（用于优化器分组）
        """
        params = [
            {'params': self.hash_encoding.parameters(), 'lr': 1e-3},  # 哈希编码
            {'params': self.tiny_mlp.parameters(), 'lr': 1e-3},       # Tiny MLP
        ]
        
        if self.region_embedding is not None:
            params.append({
                'params': self.region_embedding.parameters(), 'lr': 1e-3
            })
        
        return params


class LayoutInvariantMaskGenerator:
    """
    布局不变掩码生成器（SAM 占位）
    """
    def __init__(self):
        self.sam_model = None
        print("[LayoutInvariantMask] 初始化完成，SAM 接口待完善")
    
    def generate_mask(
        self,
        old_image: torch.Tensor,
        new_image: torch.Tensor,
        region_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        生成布局不变掩码（占位实现）
        """
        diff = torch.abs(old_image - new_image).mean(dim=0)  # [H, W]
        mask = diff < 0.1  # 阈值可调整
        return mask


class RemovalFactor(nn.Module):
    """
    移除因子（用于标记消失的高斯）
    """
    def __init__(self, num_gaussians: int):
        super().__init__()
        self.removal_factors = nn.Parameter(torch.zeros(num_gaussians, 1))
    
    def get_removal_prob(self) -> torch.Tensor:
        return torch.sigmoid(1000 * self.removal_factors)
    
    def apply(self, opacities: torch.Tensor) -> torch.Tensor:
        removal_prob = self.get_removal_prob()
        return opacities * (1 - removal_prob)
    
    def get_regularization_loss(self) -> torch.Tensor:
        prob = self.get_removal_prob()
        reg_loss = torch.mean((1 - prob) * prob)
        bce_loss = torch.mean(prob * torch.log(prob + 1e-6) + (1 - prob) * torch.log(1 - prob + 1e-6))
        return -0.01 * reg_loss + 0.001 * bce_loss


class ContinuousUpdateManager:
    """
    持续更新管理器（用于协调三阶段更新）
    """
    def __init__(
        self,
        num_regions: int = 65,
        n_offsets: int = 5,
    ):
        self.num_regions = num_regions
        
        # 全局高斯 MLP（替代区域 MLP）
        self.global_gaussian_mlp = GlobalGaussianMLP(
            num_regions=num_regions,
            n_offsets=n_offsets,
        )
        
        # 布局不变掩码生成器
        self.mask_generator = LayoutInvariantMaskGenerator()
        
        # 移除因子
        self.removal_factor = None
        
        # 当前时间戳
        self.current_timestamp = 0.0
        
        # 三阶段配置
        self.stage_configs = {
            'appearance': {'iterations': 7000},
            'geometry': {'iterations': 8000},
            'refinement': {'iterations': 15000},
        }
        
        # 当前阶段
        self.current_stage = 'appearance'
        self.stage_iteration = 0
        
        print("[ContinuousUpdateManager] 初始化完成")
    
    def get_global_mlp(self) -> GlobalGaussianMLP:
        return self.global_gaussian_mlp
    
    def set_timestamp(self, timestamp: float):
        self.current_timestamp = timestamp
    
    def init_timestamp(self, num_gaussians: int):
        """初始化新时间戳"""
        self.removal_factor = RemovalFactor(num_gaussians)
        self.current_stage = 'appearance'
        self.stage_iteration = 0
    
    def update_stage(self):
        """更新阶段"""
        self.stage_iteration += 1
        
        if self.current_stage == 'appearance' and self.stage_iteration >= self.stage_configs['appearance']['iterations']:
            self.current_stage = 'geometry'
            self.stage_iteration = 0
            print("[ContinuousUpdate] 切换到几何更新阶段")
        elif self.current_stage == 'geometry' and self.stage_iteration >= self.stage_configs['geometry']['iterations']:
            self.current_stage = 'refinement'
            self.stage_iteration = 0
            print("[ContinuousUpdate] 切换到联合精炼阶段")
    
    def get_stage(self) -> str:
        return self.current_stage
