#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import time
from datetime import timedelta
import torch
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
from einops import repeat
import math

# 导入我们创建的AnchorFeatureStorage
from anchor_feature_storage import AnchorFeatureStorage

# ====================== 新增：引入分区训练依赖 ======================
from shapely.geometry import Polygon, Point
# =====================================================================
    
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 fork: int=2,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 add_level: bool = False,
                 visible_threshold: float = -1,
                 dist2level: str = 'round',
                 base_layer: int = 10,
                 progressive: bool = True,
                 extend: float = 1.1,
                 freeze_gaussians: bool = False, # 新增：冻结高斯球开关
                 num_regions: int = 65 # 新增：区域数量
                 ):

        self.feat_dim = feat_dim
        self.view_dim = 3
        self.n_offsets = n_offsets
        self.fork = fork
        self.use_feat_bank = use_feat_bank
        self.freeze_gaussians = freeze_gaussians  # 绑定冻结标志

        self.num_regions = num_regions
        self.appearance_dim = appearance_dim
        self.embedding_appearance = nn.ModuleList()
        for i in range(self.num_regions):
            self.embedding_appearance.append(None)
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist
        self.add_level = add_level
        self.progressive = progressive
        

        # Octree
        self.sub_pos_offsets = torch.tensor([[i % fork, (i // fork) % fork, i // (fork * fork)] for i in range(fork**3)]).float().cuda()
        self.extend = extend
        self.visible_threshold = visible_threshold
        self.dist2level = dist2level
        self.base_layer = base_layer
        
        self.start_step = 0
        self.end_step = 0

        self._anchor = torch.empty(0)
        self._level = torch.empty(0)
        self._region = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self.opacity_accum = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.level_dim = 1 if self.add_level else 0
    
        # 初始化区域专家MLP
        self.mlp_opacity = nn.ModuleList()
        self.mlp_cov = nn.ModuleList()
        self.mlp_color = nn.ModuleList()
        
        for i in range(self.num_regions):
            # 不透明度MLP
            mlp_opacity = nn.Sequential(
                    nn.Linear(self.feat_dim+self.view_dim+self.opacity_dist_dim+self.level_dim, self.feat_dim),
                    nn.ReLU(True),
                    nn.Linear(self.feat_dim, self.n_offsets),
                    nn.Tanh()
                ).cuda()
            self.mlp_opacity.append(mlp_opacity)
            
            # 协方差MLP
            mlp_cov = nn.Sequential(
                    nn.Linear(self.feat_dim+self.view_dim+self.cov_dist_dim+self.level_dim, self.feat_dim),
                    nn.ReLU(True),
                    nn.Linear(self.feat_dim, 7*self.n_offsets),
                ).cuda()
            self.mlp_cov.append(mlp_cov)
            
            # 颜色MLP
            mlp_color = nn.Sequential(
                    nn.Linear(self.feat_dim+self.view_dim+self.color_dist_dim+self.level_dim+self.appearance_dim, self.feat_dim),
                    nn.ReLU(True),
                    nn.Linear(self.feat_dim, 3*self.n_offsets),
                    nn.Sigmoid()
                ).cuda()
            self.mlp_color.append(mlp_color)
        
        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                    nn.Linear(self.view_dim+self.level_dim, self.feat_dim),
                    nn.ReLU(True),
                    nn.Linear(self.feat_dim, 3),
                    nn.Softmax(dim=1)
                ).cuda()

        # ====================== 新增：分区训练相关成员变量 ======================
        self._stored_anchors = {}  # 存储区域外的锚点数据 {region_key: {data_dict}}
        self._current_region_polygon = None
        self._current_region_key = None
        # =========================================================================

    def eval(self):
        for mlp in self.mlp_opacity:
            mlp.eval()
        for mlp in self.mlp_cov:
            mlp.eval()
        for mlp in self.mlp_color:
            mlp.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()

    def train(self):
        for mlp in self.mlp_opacity:
            mlp.train()
        for mlp in self.mlp_cov:
            mlp.train()
        for mlp in self.mlp_color:
            mlp.train()
        if self.use_feat_bank:
            self.mlp_feature_bank.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()

    def capture(self):
        return (
            self._anchor,
            self._level,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        # 注意：如果是从 Checkpoint 恢复，可能需要同时恢复 _stored_anchors，
        # 但为了简单，这里假设恢复时是从头开始或不需要恢复旧区域数据。
        (self.active_sh_degree, 
        self._anchor, 
        self._level,
        self._offset,
        self._local,
        self._scaling, 
        self._rotation, 
        self._opacity,
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_appearance(self):
        def _get_appearance(camera_indicies, region=0):
            if self.appearance_dim > 0:
                return self.embedding_appearance[region](camera_indicies)
            else:
                return None
        return _get_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def get_level(self):
        return self._level
    
    @property
    def get_extra_level(self):
        return self._extra_level
        
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_anchor_feat(self):
        return self._anchor_feat
    
    def get_opacity_mlp(self, region=0):
        return self.mlp_opacity[region]   

    def get_cov_mlp(self, region=0):
        return self.mlp_cov[region]
    
    def get_color_mlp(self, region=0):
        return self.mlp_color[region]
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            for i in range(self.num_regions):
                self.embedding_appearance[i] = Embedding(num_cameras, self.appearance_dim).cuda()
        
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)    

    def set_coarse_interval(self, coarse_iter, coarse_factor):
        self.coarse_intervals = []
        num_level = self.levels - 1 - self.init_level
        if num_level > 0:
            q = 1/coarse_factor
            a1 = coarse_iter*(1-q)/(1-q**num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q ** i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)

    def set_level(self, points, cameras, scales, dist_ratio=0.95, init_level=-1, levels=-1):
        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()
        for scale in scales:
            for cam in cameras[scale]:
                cam_center = cam.camera_center
                cam_info = torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale]).float().cuda()
                self.cam_infos = torch.cat((self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)
                dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1))
                dist_max = torch.quantile(dist, dist_ratio)
                dist_min = torch.quantile(dist, 1 - dist_ratio)
                new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
                new_dist = new_dist * scale
                all_dist = torch.cat((all_dist, new_dist), dim=0)
        dist_max = torch.quantile(all_dist, dist_ratio)
        dist_min = torch.quantile(all_dist, 1 - dist_ratio)
        self.standard_dist = dist_max
        if levels == -1:
            self.levels = torch.round(torch.log2(dist_max/dist_min)/math.log2(self.fork)).int().item() + 1
        else:
            self.levels = levels
        if init_level == -1:
            self.init_level = int(self.levels/2)
        else:
            self.init_level = init_level
            
    def octree_sample(self, data, init_pos):
        torch.cuda.synchronize(); t0 = time.time()
        self.positions = torch.empty(0, 3).float().cuda()
        self._level = torch.empty(0).int().cuda() 
        for cur_level in range(self.levels):
            cur_size = self.voxel_size/(float(self.fork) ** cur_level)
            new_positions = torch.unique(torch.round((data - init_pos) / cur_size), dim=0) * cur_size + init_pos
            new_level = torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda") * cur_level
            self.positions = torch.concat((self.positions, new_positions), dim=0)
            self._level = torch.concat((self._level, new_level), dim=0)
        torch.cuda.synchronize(); t1 = time.time()
        time_diff = t1 - t0
        print(f"Building octree time: {int(time_diff // 60)} min {time_diff % 60} sec")

    def create_from_pcd(self, points, spatial_lr_scale, logger=None):
        self.spatial_lr_scale = spatial_lr_scale
        box_min = torch.min(points)*self.extend
        box_max = torch.max(points)*self.extend
        box_d = box_max - box_min
        if self.base_layer < 0:
            default_voxel_size = 0.02
            self.base_layer = torch.round(torch.log2(box_d/default_voxel_size)).int().item()-(self.levels//2)+1
        self.voxel_size = box_d/(float(self.fork) ** self.base_layer)
        self.init_pos = torch.tensor([box_min, box_min, box_min]).float().cuda()
        self.octree_sample(points, self.init_pos)

        if self.visible_threshold < 0:
            self.visible_threshold = 0.0
            self.positions, self._level, self.visible_threshold, _ = self.weed_out(self.positions, self._level)
        self.positions, self._level, _, _ = self.weed_out(self.positions, self._level)

        print(f'Branches of Tree: {self.fork}')
        print(f'Base Layer of Tree: {self.base_layer}')
        print(f'Visible Threshold: {self.visible_threshold}')
        print(f'Appearance Embedding Dimension: {self.appearance_dim}') 
        print(f'LOD Levels: {self.levels}')
        print(f'Initial Levels: {self.init_level}')
        print(f'Initial Voxel Number: {self.positions.shape[0]}')
        print(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')
        print(f'Max Voxel Size: {self.voxel_size}')
        if logger:
            logger.info(f'Branches of Tree: {self.fork}')
            logger.info(f'Base Layer of Tree: {self.base_layer}')
            logger.info(f'Visible Threshold: {self.visible_threshold}')
            logger.info(f'Appearance Embedding Dimension: {self.appearance_dim}')
            logger.info(f'LOD Levels: {self.levels}')
            logger.info(f'Initial Levels: {self.init_level}')
            logger.info(f'Initial Voxel Number: {self.positions.shape[0]}')
            logger.info(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')
            logger.info(f'Max Voxel Size: {self.voxel_size}')

        offsets = torch.zeros((self.positions.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((self.positions.shape[0], self.feat_dim)).float().cuda()
        dist2 = torch.clamp_min(distCUDA2(self.positions).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        rots = torch.zeros((self.positions.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((self.positions.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(self.positions.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self._level = self._level.unsqueeze(dim=1)
        self._extra_level = torch.zeros(self._anchor.shape[0], dtype=torch.float, device="cuda")
        self._region = torch.zeros(self._anchor.shape[0], dtype=torch.float, device="cuda")
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")

    def map_to_int_level(self, pred_level, cur_level):
        if self.dist2level=='floor':
            int_level = torch.floor(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='round':
            int_level = torch.round(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='ceil':
            int_level = torch.ceil(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='progressive':
            pred_level = torch.clamp(pred_level+1.0, min=0.9999, max=cur_level + 0.9999)
            int_level = torch.floor(pred_level).int()
            self._prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)
            self.transition_mask = (self._level.squeeze(dim=1) == int_level)
        else:
            raise ValueError(f"Unknown dist2level: {self.dist2level}")
        
        return int_level

    def weed_out(self, anchor_positions, anchor_levels):
        visible_count = torch.zeros(anchor_positions.shape[0], dtype=torch.int, device="cuda")
        for cam in self.cam_infos:
            cam_center, scale = cam[:3], cam[3]
            dist = torch.sqrt(torch.sum((anchor_positions - cam_center)**2, dim=1)) * scale
            pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork)   
            int_level = self.map_to_int_level(pred_level, self.levels - 1)
            visible_count += (anchor_levels <= int_level).int()
        visible_count = visible_count/len(self.cam_infos)
        weed_mask = (visible_count > self.visible_threshold)
        mean_visible = torch.mean(visible_count)
        return anchor_positions[weed_mask], anchor_levels[weed_mask], mean_visible, weed_mask

    def set_anchor_mask(self, cam_center, iteration, resolution_scale):
        anchor_pos = self._anchor + (self.voxel_size/2) / (float(self.fork) ** self._level)
        dist = torch.sqrt(torch.sum((anchor_pos - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        
        is_training = self.get_color_mlp(0).training
        if self.progressive and is_training:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels

        int_level = self.map_to_int_level(pred_level, coarse_index - 1)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)

    def set_anchor_mask_perlevel(self, cam_center, resolution_scale, cur_level):
        anchor_pos = self._anchor + (self.voxel_size/2) / (float(self.fork) ** self._level)
        dist = torch.sqrt(torch.sum((anchor_pos - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        int_level = self.map_to_int_level(pred_level, cur_level)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        if not self.freeze_gaussians:
            self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
            self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
            self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
            self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        
        l = []
        # 非冻结模式：仅添加高斯球参数（不再重复添加MLP）
        if not self.freeze_gaussians:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            ]
        
        # 无论是否冻结，统一且仅添加一次 MLP / Embedding 参数
        for i in range(self.num_regions):
            l.extend([
                {'params': self.mlp_opacity[i].parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": f"mlp_opacity_{i}"},
                {'params': self.mlp_cov[i].parameters(), 'lr': training_args.mlp_cov_lr_init, "name": f"mlp_cov_{i}"},
                {'params': self.mlp_color[i].parameters(), 'lr': training_args.mlp_color_lr_init, "name": f"mlp_color_{i}"},
            ])

        if self.appearance_dim > 0:
            l.append({'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"})
        if self.use_feat_bank:
            l.append({'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        # 冻结模式：不加载高斯的学习率调度器
        if not self.freeze_gaussians:
            self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)
            self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.offset_lr_delay_mult,
                                                        max_steps=training_args.offset_lr_max_steps)
        # MLP调度器始终保留    
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:

            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr    
            if param_group["name"].startswith("mlp_opacity"):
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"].startswith("mlp_cov"):
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"].startswith("mlp_color"):
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            
    def construct_list_of_attributes(self):
        l = []
        l.append('x')
        l.append('y')
        l.append('z')
        l.append('level')
        l.append('extra_level')
        l.append('region')
        l.append('info')
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        levels = self._level.detach().cpu().numpy()
        extra_levels = self._extra_level.unsqueeze(dim=1).detach().cpu().numpy()
        regions = self._region.unsqueeze(dim=1).detach().cpu().numpy()
        infos = np.zeros_like(levels, dtype=np.float32)
        infos[0, 0] = self.voxel_size
        infos[1, 0] = self.standard_dist

        anchor_feats = self._anchor_feat.detach().cpu().numpy()
        offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rots = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, levels, extra_levels, regions, infos, offsets, anchor_feats, opacities, scales, rots), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def plot_levels(self):
        for level in range(self.levels):
            level_mask = (self._level == level).squeeze(dim=1)
            print(f'Level {level}: {torch.sum(level_mask).item()}, Ratio: {torch.sum(level_mask).item()/self._level.shape[0]}')

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        
        levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis].astype(np.int32)
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
        
        # 加载region属性
        regions = np.asarray(plydata.elements[0]["region"])[... ,np.newaxis].astype(np.float32)
        
        self.voxel_size = torch.tensor(plydata.elements[0]["info"][0]).float()
        self.standard_dist = torch.tensor(plydata.elements[0]["info"][1]).float()

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
        self._extra_level = torch.tensor(extra_levels, dtype=torch.float, device="cuda").squeeze(dim=1)
        self._region = torch.tensor(regions, dtype=torch.float, device="cuda").squeeze(dim=1)
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        self.levels = torch.max(self._level) - torch.min(self._level) + 1

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # ====================== 新增：核心分区训练 API ======================
    
    def _torch_point_in_polygon(self, points: torch.Tensor, polygon: Polygon) -> torch.Tensor:
        """
        纯PyTorch实现2D点是否在多边形内（GPU加速），使用 XZ 平面作为地面
        """
        # 提取多边形顶点 (xz 平面)
        poly_coords = list(polygon.exterior.coords)
        # Shapely 出来的是 (x, y)，我们把它当作 (x, z)
        poly_tensor = torch.tensor(poly_coords, dtype=torch.float32, device=points.device)
        
        # 取点的 xz 坐标
        pts = points[:, [0, 2]]  # [N, 2]
        
        N, M = pts.shape[0], poly_tensor.shape[0]
        inside = torch.zeros(N, dtype=torch.bool, device=pts.device)
        
        x, y = pts[:, 0], pts[:, 1]
        for i in range(M):
            j = (i + 1) % M
            xi, yi = poly_tensor[i, 0], poly_tensor[i, 1]
            xj, yj = poly_tensor[j, 0], poly_tensor[j, 1]
            
            cond1 = (yi > y) != (yj > y)
            denom = yj - yi + 1e-8
            x_intersect = ((y - yi) * (xj - xi)) / denom + xi
            inside[cond1 & (x < x_intersect)] = ~inside[cond1 & (x < x_intersect)]
            
        return inside

    # def enter_region(self, region_polygon: Polygon):
    #     """
    #     进入新区域：
    #     1. 计算哪些 Anchor 在区域内
    #     2. 深度拷贝区域外的 Anchor 到 CPU 内存 (或 GPU 内存)
    #     3. **彻底删除**区域外的 Anchor
    #     """
    #     self._current_region_polygon = region_polygon
    #     region_key = str(id(region_polygon))
    #     self._current_region_key = region_key
        
    #     print(f"  [OctreeGS] Entering region...")
        
    #     # 1. 计算 Anchor 中心 (考虑八叉树层级偏移，取体素中心)
    #     anchor_centers = self._anchor + (self.voxel_size / 2) / (float(self.fork) ** self._level)
        
    #     # 2. 判断是否在区域内
    #     inside_mask = self._torch_point_in_polygon(anchor_centers, region_polygon)
    #     outside_mask = ~inside_mask
        
    #     print(f"  [OctreeGS] Anchors inside region: {inside_mask.sum().item()} / {len(inside_mask)}")

    #     # 3. 存储区域外的 Anchor 数据 (以便后续合并或恢复)
    #     if outside_mask.sum() > 0:
    #         print(f"  [OctreeGS] Storing {outside_mask.sum().item()} anchors outside region...")
    #         # 存储所有必要的张量，包括统计量
    #         self._stored_anchors[region_key] = {
    #             "_anchor": self._anchor.data[outside_mask].clone(),
    #             "_level": self._level[outside_mask].clone(),
    #             "_extra_level": self._extra_level[outside_mask].clone(),
    #             "_region": self._region[outside_mask].clone(),
    #             "_offset": self._offset.data[outside_mask].clone(),
    #             "_anchor_feat": self._anchor_feat.data[outside_mask].clone(),
    #             "_scaling": self._scaling.data[outside_mask].clone(),
    #             "_rotation": self._rotation.data[outside_mask].clone(),
    #             "_opacity": self._opacity.data[outside_mask].clone(),
    #             # 同时存储训练统计量，虽然区域独立一般不需要恢复统计量，但存下来以防万一
    #             "opacity_accum": self.opacity_accum[outside_mask].clone() if self.opacity_accum.numel() > 0 else None,
    #             "anchor_demon": self.anchor_demon[outside_mask].clone() if self.anchor_demon.numel() > 0 else None,
    #         }
            
    #         # 4. **删除**区域外的 Anchor
    #         print(f"  [OctreeGS] Deleting anchors outside region...")
    #         self.prune_anchor(outside_mask)
    #         print(f"  [OctreeGS] Remaining anchors: {self._anchor.shape[0]}")
    #     else:
    #         print(f"  [OctreeGS] No anchors outside region to store/delete.")

    def enter_region(self, region_polygon: Polygon):
        """
        进入新区域：
        1. 保存所有高斯球（仅当未保存过时）
        2. 不删除任何高斯球
        """
        self._current_region_polygon = region_polygon
        region_key = str(id(region_polygon))
        self._current_region_key = region_key
        
        print(f"  [OctreeGS] Entering region...")
        
        # 只有当 _stored_anchors 为空时才保存
        if not self._stored_anchors:
            # 保存所有高斯球数据
            print(f"  [OctreeGS] Saving all {self._anchor.shape[0]} anchors...")
            # 存储所有必要的张量，包括统计量
            self._stored_anchors = {
                "_anchor": self._anchor.data.clone(),
                "_level": self._level.clone(),
                "_extra_level": self._extra_level.clone(),
                "_region": self._region.clone(),
                "_offset": self._offset.data.clone(),
                "_anchor_feat": self._anchor_feat.data.clone(),
                "_scaling": self._scaling.data.clone(),
                "_rotation": self._rotation.data.clone(),
                "_opacity": self._opacity.data.clone(),
                # 同时存储训练统计量
                "opacity_accum": self.opacity_accum.clone() if self.opacity_accum.numel() > 0 else None,
                "anchor_demon": self.anchor_demon.clone() if self.anchor_demon.numel() > 0 else None,
            }
            
            print(f"  [OctreeGS] All anchors saved successfully")
        else:
            print(f"  [OctreeGS] Anchors already saved, skipping save")
    def exit_and_cleanup(self):
        """
        离开区域：加载之前保存的所有高斯球
        """
        if not self._stored_anchors:
            return
            
        stored_data = self._stored_anchors
        
        if stored_data is not None and len(stored_data["_anchor"]) > 0:
            print(f"  [OctreeGS] Restoring {len(stored_data['_anchor'])} stored anchors...")
            
            # 构建恢复字典
            restore_dict = {
                "anchor": stored_data["_anchor"],
                "offset": stored_data["_offset"],
                "anchor_feat": stored_data["_anchor_feat"],
                "scaling": stored_data["_scaling"],
                "rotation": stored_data["_rotation"],
                "opacity": stored_data["_opacity"],
            }
            
            # 调用 OctreeGS 原有的拼接方法
            optimizable_tensors = self.cat_tensors_to_optimizer(restore_dict)
            
            # 更新参数
            self._anchor = optimizable_tensors["anchor"]
            self._offset = optimizable_tensors["offset"]
            self._anchor_feat = optimizable_tensors["anchor_feat"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
            self._opacity = optimizable_tensors["opacity"]
            
            # 拼接 level、extra_level 和 region
            self._level = torch.cat([self._level, stored_data["_level"]], dim=0)
            self._extra_level = torch.cat([self._extra_level, stored_data["_extra_level"]], dim=0)
            self._region = torch.cat([self._region, stored_data["_region"]], dim=0)
            
            # 扩展统计量 (简单补零，因为区域独立，之前的统计量不重要)
            num_new = len(stored_data["_anchor"])
            self.opacity_accum = torch.cat([self.opacity_accum, torch.zeros(num_new, 1, device='cuda')], dim=0)
            self.anchor_demon = torch.cat([self.anchor_demon, torch.zeros(num_new, 1, device='cuda')], dim=0)
            
            # offset 相关的统计量需要扩展 n_offsets 倍
            num_new_offsets = num_new * self.n_offsets
            # 检查是否已经初始化，防止报错
            if not hasattr(self, 'offset_gradient_accum') or self.offset_gradient_accum.numel() == 0:
                 # 如果不存在，先初始化为当前长度
                 cur_len = self._anchor.shape[0] * self.n_offsets - num_new_offsets
                 self.offset_gradient_accum = torch.zeros(cur_len, 1, device='cuda')
                 self.offset_denom = torch.zeros(cur_len, 1, device='cuda')
            
            self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, torch.zeros(num_new_offsets, 1, device='cuda')], dim=0)
            self.offset_denom = torch.cat([self.offset_denom, torch.zeros(num_new_offsets, 1, device='cuda')], dim=0)
            
            print(f"  [OctreeGS] Total anchors after restore: {self._anchor.shape[0]}")
        
        self._current_region_key = None
        self._current_region_polygon = None
    # def exit_and_cleanup(self):
    #     """
    #     离开区域：恢复之前存储的区域外锚点
    #     """
    #     if self._current_region_key is None:
    #         return
            
    #     region_key = self._current_region_key
    #     stored_data = self._stored_anchors.get(region_key, None)
        
    #     if stored_data is not None and len(stored_data["_anchor"]) > 0:
    #         print(f"  [OctreeGS] Restoring {len(stored_data['_anchor'])} stored anchors...")
            
    #         # 构建恢复字典
    #         restore_dict = {
    #             "anchor": stored_data["_anchor"],
    #             "offset": stored_data["_offset"],
    #             "anchor_feat": stored_data["_anchor_feat"],
    #             "scaling": stored_data["_scaling"],
    #             "rotation": stored_data["_rotation"],
    #             "opacity": stored_data["_opacity"],
    #         }
            
    #         # 调用 OctreeGS 原有的拼接方法
    #         optimizable_tensors = self.cat_tensors_to_optimizer(restore_dict)
            
    #         # 更新参数
    #         self._anchor = optimizable_tensors["anchor"]
    #         self._offset = optimizable_tensors["offset"]
    #         self._anchor_feat = optimizable_tensors["anchor_feat"]
    #         self._scaling = optimizable_tensors["scaling"]
    #         self._rotation = optimizable_tensors["rotation"]
    #         self._opacity = optimizable_tensors["opacity"]
            
    #         # 拼接 level、extra_level 和 region
    #         self._level = torch.cat([self._level, stored_data["_level"]], dim=0)
    #         self._extra_level = torch.cat([self._extra_level, stored_data["_extra_level"]], dim=0)
    #         self._region = torch.cat([self._region, stored_data["_region"]], dim=0)
            
    #         # 扩展统计量 (简单补零，因为区域独立，之前的统计量不重要)
    #         num_new = len(stored_data["_anchor"])
    #         self.opacity_accum = torch.cat([self.opacity_accum, torch.zeros(num_new, 1, device='cuda')], dim=0)
    #         self.anchor_demon = torch.cat([self.anchor_demon, torch.zeros(num_new, 1, device='cuda')], dim=0)
            
    #         # offset 相关的统计量需要扩展 n_offsets 倍
    #         num_new_offsets = num_new * self.n_offsets
    #         # 检查是否已经初始化，防止报错
    #         if not hasattr(self, 'offset_gradient_accum') or self.offset_gradient_accum.numel() == 0:
    #              # 如果不存在，先初始化为当前长度
    #              cur_len = self._anchor.shape[0] * self.n_offsets - num_new_offsets
    #              self.offset_gradient_accum = torch.zeros(cur_len, 1, device='cuda')
    #              self.offset_denom = torch.zeros(cur_len, 1, device='cuda')
            
    #         self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, torch.zeros(num_new_offsets, 1, device='cuda')], dim=0)
    #         self.offset_denom = torch.cat([self.offset_denom, torch.zeros(num_new_offsets, 1, device='cuda')], dim=0)
            
    #         print(f"  [OctreeGS] Total anchors after restore: {self._anchor.shape[0]}")
        
    #     # 清理存储
    #     if region_key in self._stored_anchors:
    #         del self._stored_anchors[region_key]
    #     self._current_region_key = None
        # self._current_region_polygon = None
    
        
        # 清理存储
        # self._stored_anchors = {}
        # self._current_region_polygon = None
    def set_region(self, region_id):
        """
        为当前所有高斯设置region属性
        参数:
            region_id: 区域ID，将被赋值给所有高斯的region属性
        """
        self._region.fill_(region_id)
        print(f"  [OctreeGS] Set region {region_id} for all {self._region.shape[0]} gaussians")

    def save_region_anchors(self, region_id, save_path):
        """
        存储当前区域的锚点到PLY文件
        参数:
            region_id: 区域ID
            save_path: 保存路径
        """
        import os
        import numpy as np
        from plyfile import PlyData, PlyElement
        
        # 创建保存目录
        save_dir = os.path.join(save_path, f"region_{region_id}")
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存锚点数据为PLY文件
        anchor_file = os.path.join(save_dir, "anchors.ply")
        
        anchor = self._anchor.detach().cpu().numpy()
        levels = self._level.detach().cpu().numpy()
        extra_levels = self._extra_level.unsqueeze(dim=1).detach().cpu().numpy()
        regions = self._region.unsqueeze(dim=1).detach().cpu().numpy()
        infos = np.zeros_like(levels, dtype=np.float32)
        infos[0, 0] = self.voxel_size
        infos[1, 0] = self.standard_dist

        anchor_feats = self._anchor_feat.detach().cpu().numpy()
        offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rots = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, levels, extra_levels, regions, infos, offsets, anchor_feats, opacities, scales, rots), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(anchor_file)
        
        print(f"  [OctreeGS] Saved region {region_id} anchors to {anchor_file}")

    # ====================================================================

    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        """ 冻结模式：禁止统计梯度（无需 densify）"""
        if self.freeze_gaussians:
            return
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1
    def save_initial_state(self):
        """保存初始高斯球状态"""
        self._initial_state = {
            'anchor': self._anchor.detach().clone(),
            'level': self._level.detach().clone(),
            'region': self._region.detach().clone(),
            'offset': self._offset.detach().clone(),
            'anchor_feat': self._anchor_feat.detach().clone(),
            'scaling': self._scaling.detach().clone(),
            'rotation': self._rotation.detach().clone(),
            'opacity': self._opacity.detach().clone()
        }
        print(f"[GaussianModel] 保存初始状态，锚点数量: {self._anchor.shape[0]}")   

    def restore_initial_state(self):
        """恢复初始高斯球状态"""
        if hasattr(self, '_initial_state'):
            self._anchor = nn.Parameter(self._initial_state['anchor'].requires_grad_(True))
            self._level = self._initial_state['level']
            self._region = self._initial_state['region']
            self._offset = nn.Parameter(self._initial_state['offset'].requires_grad_(True))
            self._anchor_feat = nn.Parameter(self._initial_state['anchor_feat'].requires_grad_(True))
            self._scaling = nn.Parameter(self._initial_state['scaling'].requires_grad_(True))
            self._rotation = nn.Parameter(self._initial_state['rotation'].requires_grad_(False))
            self._opacity = nn.Parameter(self._initial_state['opacity'].requires_grad_(False))
    
    def init_anchor_feature_storage(self, save_path, format='pt'):
        """
        初始化_anchor_feat存储
        参数:
            save_path: 保存路径
            format: 保存格式，'pt'或'json'
        """
        self._anchor_feature_storage = AnchorFeatureStorage(save_path, format, lazy_load=False)
        print(f"  [OctreeGS] Initialized anchor feature storage at {save_path}")
    
    def save_current_anchor_features(self, region_id, moment=0):
        """
        保存当前所有高斯球的_anchor_feat
        参数:
            region_id: 区域ID
            moment: 时刻值，默认为0
        """
        if not hasattr(self, '_anchor_feature_storage'):
            print(f"  [OctreeGS] Error: Anchor feature storage not initialized")
            return
        
        # 获取高斯球位置（考虑八叉树层级偏移，取体素中心）
        anchor_centers = self._anchor + (self.voxel_size / 2) / (float(self.fork) ** self._level)
        
        # 添加到存储
        self._anchor_feature_storage.add(
            region=region_id,
            moment=moment,
            positions=anchor_centers,
            anchor_feats=self._anchor_feat
        )
        
        # 保存到文件
        self._anchor_feature_storage.save()
        print(f"  [OctreeGS] Saved anchor features for region {region_id}, moment {moment}")
    
    def load_anchor_features_from_file(self, file_path, format='pt'):
        """
        从文件加载 anchor feature 存储
        参数:
            file_path: 文件路径
            format: 文件格式，'pt'或'json'
        """
        from anchor_feature_storage import AnchorFeatureStorage
        self._anchor_feature_storage = AnchorFeatureStorage(file_path, format, lazy_load=True)
    
    def apply_anchor_features(self, region, moment, tolerance=1e-3):
        """
        根据 region 和 moment 从存储中匹配并应用 _anchor_feat
        参数:
            region: 区域索引
            moment: 时刻值
            tolerance: 匹配容差
        """
        if not hasattr(self, '_anchor_feature_storage') or self._anchor_feature_storage is None:
            print("[GaussianModel] Error: Anchor feature storage not initialized")
            return
        
        # 获取当前高斯球的位置（考虑八叉树层级偏移）
        anchor_centers = self._anchor + (self.voxel_size / 2) / (float(self.fork) ** self._level)
        
        # 匹配特征
        assigned_feats, match_mask = self._anchor_feature_storage.match_and_assign_features(
            region, moment, anchor_centers, tolerance
        )
        
        if assigned_feats is None:
            print(f"[GaussianModel] No anchor features found for region {region}, moment {moment}")
            return
        
        # 应用匹配到的特征
        if torch.any(match_mask):
            self._anchor_feat.data[match_mask] = assigned_feats[match_mask].to(self._anchor_feat.device)
            print(f"[GaussianModel] Applied {torch.sum(match_mask)} anchor features for region {region}, moment {moment}")
        else:
            print(f"[GaussianModel] No matching anchor features found for region {region}, moment {moment}")
    
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                
                # 获取新的参数张量
                new_param_data = group["params"][0][mask].data
                
                # 【修复】对 scaling 进行 clamp，但不使用原地操作
                if group['name'] == "scaling":
                    # 分离出后3维进行 clamp，然后拼接回去
                    part1 = new_param_data[:, :3]
                    part2 = new_param_data[:, 3:]
                    # 使用 clamp 而不是原地索引赋值
                    part2_clamped = torch.clamp(part2, max=0.05)
                    new_param_data = torch.cat([part1, part2_clamped], dim=1)
                
                # 创建新的 Parameter
                group["params"][0] = nn.Parameter(new_param_data.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 获取新的参数张量
                new_param_data = group["params"][0][mask].data
                
                # 【修复】对 scaling 进行 clamp，但不使用原地操作
                if group['name'] == "scaling":
                    part1 = new_param_data[:, :3]
                    part2 = new_param_data[:, 3:]
                    part2_clamped = torch.clamp(part2, max=0.05)
                    new_param_data = torch.cat([part1, part2_clamped], dim=1)
                
                group["params"][0] = nn.Parameter(new_param_data.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
            
        return optimizable_tensors

    def prune_anchor(self, mask):
        """ 冻结模式：禁止剪枝高斯 """
        if self.freeze_gaussians:
            return
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._level = self._level[valid_points_mask]    
        self._extra_level = self._extra_level[valid_points_mask]
        self._region = self._region[valid_points_mask]
        
        # ====================== 【新增】同步剪枝训练统计量 ======================
        # 检查统计量是否已初始化且长度匹配
        if hasattr(self, 'opacity_accum') and self.opacity_accum.numel() > 0:
            if self.opacity_accum.shape[0] == mask.shape[0]:
                self.opacity_accum = self.opacity_accum[valid_points_mask]
        
        if hasattr(self, 'anchor_demon') and self.anchor_demon.numel() > 0:
            if self.anchor_demon.shape[0] == mask.shape[0]:
                self.anchor_demon = self.anchor_demon[valid_points_mask]
        
        # Offset 相关的统计量长度是 n_offsets 倍
        if hasattr(self, 'offset_gradient_accum') and self.offset_gradient_accum.numel() > 0:
            expected_len = mask.shape[0] * self.n_offsets
            if self.offset_gradient_accum.shape[0] == expected_len:
                # 将 mask 扩展 n_offsets 倍
                offset_mask = valid_points_mask.unsqueeze(1).repeat(1, self.n_offsets).view(-1)
                self.offset_gradient_accum = self.offset_gradient_accum[offset_mask]
        
        if hasattr(self, 'offset_denom') and self.offset_denom.numel() > 0:
            expected_len = mask.shape[0] * self.n_offsets
            if self.offset_denom.shape[0] == expected_len:
                offset_mask = valid_points_mask.unsqueeze(1).repeat(1, self.n_offsets).view(-1)
                self.offset_denom = self.offset_denom[offset_mask]
        # =========================================================================
    
    def get_remove_duplicates(self, grid_coords, selected_grid_coords_unique, use_chunk = True):
        if use_chunk:
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            remove_duplicates_list = []
            for i in range(max_iters):
                cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                remove_duplicates_list.append(cur_remove_duplicates)
            remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
        else:
            remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)
        return remove_duplicates
    
    def anchor_growing(self, iteration, grads, threshold, update_ratio, extra_ratio, extra_up, offset_mask):
        """ 冻结模式：禁止生长高斯 """
        if self.freeze_gaussians:
            return
        init_length = self.get_anchor.shape[0]

        # 确保 offset_mask 的大小与 grads 匹配
        if offset_mask is not None and offset_mask.shape[0] != grads.shape[0]:
            # 调整 offset_mask 的大小
            offset_mask = offset_mask[:grads.shape[0]]
            print(f"  [OctreeGS] 调整 offset_mask 大小: {offset_mask.shape[0]}")

        grads[~offset_mask] = 0.0
        anchor_grads = torch.sum(grads.reshape(-1, self.n_offsets), dim=-1) / (torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
        for cur_level in range(self.levels):
            update_value = self.fork ** update_ratio
            level_mask = (self.get_level == cur_level).squeeze(dim=1)
            level_ds_mask = (self.get_level == cur_level + 1).squeeze(dim=1)
            if torch.sum(level_mask) == 0:
                continue
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            ds_size = cur_size / self.fork
            # update threshold
            cur_threshold = threshold * (update_value ** cur_level)
            ds_threshold = cur_threshold * update_value
            extra_threshold = cur_threshold * extra_ratio
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold) & (grads < ds_threshold)
            candidate_ds_mask = (grads >= ds_threshold)
            candidate_extra_mask = (anchor_grads >= extra_threshold)

            length_inc = self.get_anchor.shape[0] - init_length
            if length_inc > 0 :
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_ds_mask = torch.cat([candidate_ds_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_extra_mask = torch.cat([candidate_extra_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)   
            
            repeated_mask = repeat(level_mask, 'n -> (n k)', k=self.n_offsets)
            candidate_mask = torch.logical_and(candidate_mask, repeated_mask)
            candidate_ds_mask = torch.logical_and(candidate_ds_mask, repeated_mask)
            if ~self.progressive or iteration > self.coarse_intervals[-1]:
                self._extra_level += extra_up * candidate_extra_mask.float()    

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)

            grid_coords = torch.round((self.get_anchor[level_mask]-self.init_pos)/cur_size).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round((selected_xyz-self.init_pos)/cur_size).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
            if selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates
                candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size+self.init_pos
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.ones([0], dtype=torch.bool, device='cuda')
                new_level = torch.zeros([0], dtype=torch.int, device='cuda')

            if (~self.progressive or iteration > self.coarse_intervals[-1]) and cur_level < self.levels - 1:
                grid_coords_ds = torch.round((self.get_anchor[level_ds_mask]-self.init_pos)/ds_size).int()
                selected_xyz_ds = all_xyz.view([-1, 3])[candidate_ds_mask]
                selected_grid_coords_ds = torch.round((selected_xyz_ds-self.init_pos)/ds_size).int()
                selected_grid_coords_unique_ds, inverse_indices_ds = torch.unique(selected_grid_coords_ds, return_inverse=True, dim=0)
                if selected_grid_coords_unique_ds.shape[0] > 0 and grid_coords_ds.shape[0] > 0:
                    remove_duplicates_ds = self.get_remove_duplicates(grid_coords_ds, selected_grid_coords_unique_ds)
                    remove_duplicates_ds = ~remove_duplicates_ds
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.weed_out(candidate_anchor_ds, new_level_ds)
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                else:
                    candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                    remove_duplicates_ds = torch.ones([0], dtype=torch.bool, device='cuda')
                    new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')
            else:
                candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates_ds = torch.ones([0], dtype=torch.bool, device='cuda')
                new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')

            if candidate_anchor.shape[0] + candidate_anchor_ds.shape[0] > 0:
                
                new_anchor = torch.cat([candidate_anchor, candidate_anchor_ds], dim=0)
                new_level = torch.cat([new_level, new_level_ds]).unsqueeze(dim=1).float().cuda()
                
                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                new_feat_ds = torch.zeros([candidate_anchor_ds.shape[0], self.feat_dim], dtype=torch.float, device='cuda')
                new_feat = torch.cat([new_feat, new_feat_ds], dim=0)
                
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling_ds = torch.ones_like(candidate_anchor_ds).repeat([1,2]).float().cuda()*ds_size # *0.05
                new_scaling = torch.cat([new_scaling, new_scaling_ds], dim=0)
                new_scaling = torch.log(new_scaling)
                
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation_ds = torch.zeros([candidate_anchor_ds.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation = torch.cat([new_rotation, new_rotation_ds], dim=0)
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))
                new_opacities_ds = inverse_sigmoid(0.1 * torch.ones((candidate_anchor_ds.shape[0], 1), dtype=torch.float, device="cuda"))
                new_opacities = torch.cat([new_opacities, new_opacities_ds], dim=0)

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                new_offsets_ds = torch.zeros_like(candidate_anchor_ds).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                new_offsets = torch.cat([new_offsets, new_offsets_ds], dim=0)

                new_extra_level = torch.zeros(candidate_anchor.shape[0], dtype=torch.float, device='cuda')
                new_extra_level_ds = torch.zeros(candidate_anchor_ds.shape[0], dtype=torch.float, device='cuda')
                new_extra_level = torch.cat([new_extra_level, new_extra_level_ds])
                
                d = {
                    "anchor": new_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }   

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                self._level = torch.cat([self._level, new_level], dim=0)
                self._extra_level = torch.cat([self._extra_level, new_extra_level], dim=0)
                # 为新锚点添加region属性，继承当前区域的region值
                if hasattr(self, '_region') and self._region.numel() > 0:
                    # 获取当前区域的region值（假设所有现有锚点属于同一区域）
                    current_region = self._region[0].item()
                    # 为新锚点创建region属性
                    new_region = torch.full((new_anchor.shape[0],), current_region, dtype=torch.float, device='cuda')
                    self._region = torch.cat([self._region, new_region], dim=0)

    def adjust_anchor(self, iteration, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, update_ratio=0.5, extra_ratio=4.0, extra_up=0.25, min_opacity=0.005):
        """ 冻结模式：禁止调整高斯密度 """
        if self.freeze_gaussians:
            return
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(iteration, grads_norm, grad_threshold, update_ratio, extra_ratio, extra_up, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        # 首先保存原始的锚点数量
        original_anchor_count = self.get_anchor.shape[0]
        
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # 检查锚点数量是否在anchor_growing中发生了变化
        current_anchor_count = self.get_anchor.shape[0]
        if current_anchor_count != original_anchor_count:
            # 如果锚点数量发生了变化，重新生成prune_mask
            # 首先扩展统计量以匹配新的锚点数量
            if hasattr(self, 'opacity_accum') and self.opacity_accum.numel() > 0:
                if self.opacity_accum.shape[0] < current_anchor_count:
                    padding = torch.zeros([current_anchor_count - self.opacity_accum.shape[0], 1], 
                                        dtype=torch.float, device='cuda')
                    self.opacity_accum = torch.cat([self.opacity_accum, padding], dim=0)
            
            if hasattr(self, 'anchor_demon') and self.anchor_demon.numel() > 0:
                if self.anchor_demon.shape[0] < current_anchor_count:
                    padding = torch.zeros([current_anchor_count - self.anchor_demon.shape[0], 1], 
                                        dtype=torch.float, device='cuda')
                    self.anchor_demon = torch.cat([self.anchor_demon, padding], dim=0)
            
            # 重新生成prune_mask
            prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
            anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
            prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0 and prune_mask.shape[0] == self.get_anchor.shape[0]:
            self.prune_anchor(prune_mask)

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.eval()
            # 检查是否是多区域MLP
            if hasattr(self, 'num_regions') and self.num_regions > 1:
                # 为每个区域的MLP单独保存
                for i in range(self.num_regions):
                    # 保存opacity MLP
                    opacity_mlp = torch.jit.trace(self.mlp_opacity[i], (torch.rand(1, self.feat_dim+self.view_dim+self.opacity_dist_dim+self.level_dim).cuda()))
                    opacity_mlp.save(os.path.join(path, f'opacity_mlp_{i}.pt'))
                    # 保存cov MLP
                    cov_mlp = torch.jit.trace(self.mlp_cov[i], (torch.rand(1, self.feat_dim+self.view_dim+self.cov_dist_dim+self.level_dim).cuda()))
                    cov_mlp.save(os.path.join(path, f'cov_mlp_{i}.pt'))
                    # 保存color MLP
                    color_mlp = torch.jit.trace(self.mlp_color[i], (torch.rand(1, self.feat_dim+self.view_dim+self.color_dist_dim+self.appearance_dim+self.level_dim).cuda()))
                    color_mlp.save(os.path.join(path, f'color_mlp_{i}.pt'))
                    # 保存外观编码
                    if self.appearance_dim > 0 and self.embedding_appearance[i] is not None:
                        emd = torch.jit.trace(self.embedding_appearance[i], (torch.zeros((1,), dtype=torch.long).cuda()))
                        emd.save(os.path.join(path, f'embedding_appearance_{i}.pt'))
            else:
                # 单区域情况
                opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+self.view_dim+self.opacity_dist_dim+self.level_dim).cuda()))
                opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
                cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+self.view_dim+self.cov_dist_dim+self.level_dim).cuda()))
                cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
                color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+self.view_dim+self.color_dist_dim+self.appearance_dim+self.level_dim).cuda()))
                color_mlp.save(os.path.join(path, 'color_mlp.pt'))
                if self.appearance_dim > 0:
                    emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                    emd.save(os.path.join(path, 'embedding_appearance.pt'))
            # 保存其他不变的MLP
            if self.use_feat_bank:
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+self.level_dim).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
            self.train()
        elif mode == 'unite':
            param_dict = {}
            param_dict['opacity_mlp'] = [mlp.state_dict() for mlp in self.mlp_opacity]
            param_dict['cov_mlp'] = [mlp.state_dict() for mlp in self.mlp_cov]
            param_dict['color_mlp'] = [mlp.state_dict() for mlp in self.mlp_color]
            if self.appearance_dim > 0:
                param_dict['appearance'] = [emb.state_dict() for emb in self.embedding_appearance]
            if self.use_feat_bank:
                param_dict['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
            torch.save(param_dict, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            # 检查是否是多区域MLP
            if hasattr(self, 'num_regions') and self.num_regions > 1:
                # 为每个区域加载MLP
                from torch.nn import ModuleList
                self.mlp_opacity = ModuleList()
                self.mlp_cov = ModuleList()
                self.mlp_color = ModuleList()
                self.embedding_appearance = ModuleList()
                for i in range(self.num_regions):
                    # 加载opacity MLP
                    self.mlp_opacity.append(torch.jit.load(os.path.join(path, f'opacity_mlp_{i}.pt')).cuda())
                    # 加载cov MLP
                    self.mlp_cov.append(torch.jit.load(os.path.join(path, f'cov_mlp_{i}.pt')).cuda())
                    # 加载color MLP
                    self.mlp_color.append(torch.jit.load(os.path.join(path, f'color_mlp_{i}.pt')).cuda())
                    # 加载外观编码
                    if self.appearance_dim > 0:
                        self.embedding_appearance.append(torch.jit.load(os.path.join(path, f'embedding_appearance_{i}.pt')).cuda())
                    else:
                        self.embedding_appearance.append(None)
            else:
                # 单区域情况
                self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
                self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
                self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
                if self.appearance_dim > 0:
                    self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
            # 加载其他不变的MLP
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            for i, state_dict in enumerate(checkpoint['opacity_mlp']):
                self.mlp_opacity[i].load_state_dict(state_dict)
            for i, state_dict in enumerate(checkpoint['cov_mlp']):
                self.mlp_cov[i].load_state_dict(state_dict)
            for i, state_dict in enumerate(checkpoint['color_mlp']):
                self.mlp_color[i].load_state_dict(state_dict)
            if self.appearance_dim > 0 and 'appearance' in checkpoint:
                for i, state_dict in enumerate(checkpoint['appearance']):
                    self.embedding_appearance[i].load_state_dict(state_dict)
    
    def load_mlp_from_pt(self, path):
        """
        从PT文件加载MLP参数
        参数:
            path: PT文件路径
        """
        import torch
        from torch.nn import ModuleList
        
        # 加载MLP状态字典
        mlp_state_dict = torch.load(path)
        
        # 检查是否是多区域MLP
        if 'mlp_opacity' in mlp_state_dict and isinstance(mlp_state_dict['mlp_opacity'], list):
            # 多区域情况
            num_regions = len(mlp_state_dict['mlp_opacity'])
            self.num_regions = num_regions
            
            # 重新创建MLP模块
            self.mlp_opacity = ModuleList()
            self.mlp_cov = ModuleList()
            self.mlp_color = ModuleList()
            
            # 为每个区域创建并加载MLP（使用与保存的模型相同的结构）
            for i in range(num_regions):
                # 从保存的模型中获取输入输出维度
                opacity_state = mlp_state_dict['mlp_opacity'][i]
                cov_state = mlp_state_dict['mlp_cov'][i]
                color_state = mlp_state_dict['mlp_color'][i]
                
                # 获取输入维度
                opacity_in_dim = opacity_state['0.weight'].shape[1]
                cov_in_dim = cov_state['0.weight'].shape[1]
                color_in_dim = color_state['0.weight'].shape[1]
                
                # 获取输出维度
                opacity_out_dim = opacity_state['2.weight'].shape[0]
                cov_out_dim = cov_state['2.weight'].shape[0]
                color_out_dim = color_state['2.weight'].shape[0]
                
                # 强制更新n_offsets为保存的模型值
                self.n_offsets = opacity_out_dim
                print(f"⚠️  从MLP文件中强制更新n_offsets为: {self.n_offsets}")
                
                # 创建新的MLP - 不透明度
                mlp_opacity = nn.Sequential(
                    nn.Linear(opacity_in_dim, self.feat_dim),
                    nn.ReLU(True),
                    nn.Linear(self.feat_dim, opacity_out_dim),
                    nn.Tanh()
                ).cuda()
                
                # 创建新的MLP - 协方差
                mlp_cov = nn.Sequential(
                    nn.Linear(cov_in_dim, self.feat_dim),
                    nn.ReLU(True),
                    nn.Linear(self.feat_dim, cov_out_dim),
                ).cuda()
                
                # 创建新的MLP - 颜色
                mlp_color = nn.Sequential(
                    nn.Linear(color_in_dim, self.feat_dim),
                    nn.ReLU(True),
                    nn.Linear(self.feat_dim, color_out_dim),
                    nn.Sigmoid()
                ).cuda()
                
                # 加载参数
                mlp_opacity.load_state_dict(opacity_state)
                mlp_cov.load_state_dict(cov_state)
                mlp_color.load_state_dict(color_state)
                
                # 添加到模块列表
                self.mlp_opacity.append(mlp_opacity)
                self.mlp_cov.append(mlp_cov)
                self.mlp_color.append(mlp_color)
        else:
            # 单区域情况（使用与保存的模型相同的结构）
            # 从保存的模型中获取输入输出维度
            opacity_state = mlp_state_dict['mlp_opacity']
            cov_state = mlp_state_dict['mlp_cov']
            color_state = mlp_state_dict['mlp_color']
            
            # 获取输入维度
            opacity_in_dim = opacity_state['0.weight'].shape[1]
            cov_in_dim = cov_state['0.weight'].shape[1]
            color_in_dim = color_state['0.weight'].shape[1]
            
            # 获取输出维度
            opacity_out_dim = opacity_state['2.weight'].shape[0]
            cov_out_dim = cov_state['2.weight'].shape[0]
            color_out_dim = color_state['2.weight'].shape[0]
            
            # 强制更新n_offsets为保存的模型值
            self.n_offsets = opacity_out_dim
            print(f"⚠️  从MLP文件中强制更新n_offsets为: {self.n_offsets}")
            
            # 创建新的MLP - 不透明度
            self.mlp_opacity = nn.Sequential(
                nn.Linear(opacity_in_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, opacity_out_dim),
                nn.Tanh()
            ).cuda()
            
            # 创建新的MLP - 协方差
            self.mlp_cov = nn.Sequential(
                nn.Linear(cov_in_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, cov_out_dim),
            ).cuda()
            
            # 创建新的MLP - 颜色
            self.mlp_color = nn.Sequential(
                nn.Linear(color_in_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, color_out_dim),
                nn.Sigmoid()
            ).cuda()
            
            # 加载参数
            self.mlp_opacity.load_state_dict(opacity_state)
            self.mlp_cov.load_state_dict(cov_state)
            self.mlp_color.load_state_dict(color_state)
        
        # 加载其他不变的MLP
        if self.use_feat_bank and 'mlp_feature_bank' in mlp_state_dict:
            # 使用与保存的模型相同的结构
            feature_bank_state = mlp_state_dict['mlp_feature_bank']
            in_dim = feature_bank_state['0.weight'].shape[1]
            out_dim = feature_bank_state['2.weight'].shape[0]
            
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(in_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, out_dim),
                nn.Softmax(dim=1)
            ).cuda()
            self.mlp_feature_bank.load_state_dict(feature_bank_state)
        
        if self.appearance_dim > 0 and 'embedding_appearance' in mlp_state_dict:
            # 保持原样
            self.embedding_appearance = Embedding(self.embedding_appearance.num_embeddings, self.embedding_appearance.embedding_dim) if hasattr(self, 'embedding_appearance') else Embedding(1, self.appearance_dim)
            self.embedding_appearance.load_state_dict(mlp_state_dict['embedding_appearance'])
            self.embedding_appearance = self.embedding_appearance.cuda()
    def clean_in_region(self):
        """
        彻底清理当前激活区域内的所有高斯（训练完成后清空）
        """
        if self._current_region_name is None:
            print("⚠️  当前无激活区域，跳过清理")
            return
            
        print(f"\n🗑️  [CleanInRegion] 清理当前区域【{self._current_region_name}】内的所有高斯...")
        
        # 构建全True的mask（删除所有）
        full_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device='cuda')
        self.prune_anchor(full_mask)
        
        # 重置区域状态
        self._current_region_polygon = None
        self._current_region_inside_mask = None
        self._current_region_name = None
        
        # 清空统计量
        self.opacity_accum = torch.empty(0, device='cuda')
        self.offset_gradient_accum = torch.empty(0, device='cuda')
        self.offset_denom = torch.empty(0, device='cuda')
        self.anchor_demon = torch.empty(0, device='cuda')
        
        print(f"✅ 清理完成，当前高斯数量: {self._anchor.shape[0]}")

    def clean_out_region(self, region_polygon: Polygon, region_name: str):
        """
        彻底清理区域外的所有高斯（只保留区域内）
        """
        self._current_region_polygon = region_polygon
        self._current_region_name = region_name
        
        print(f"\n🗑️  [CleanOutRegion] 清理区域【{region_name}】外的高斯...")
        
        # 1. 计算 Anchor 中心 (考虑八叉树层级偏移，取体素中心)
        anchor_centers = self._anchor + (self.voxel_size / 2) / (float(self.fork) ** self._level)
        
        # 2. 判断是否在区域内
        self._current_region_inside_mask = self._torch_point_in_polygon(anchor_centers, region_polygon)
        outside_mask = ~self._current_region_inside_mask
        
        print(f"   🔍 区域内高斯数量: {self._current_region_inside_mask.sum().item()}")
        print(f"   🔍 区域外高斯数量: {outside_mask.sum().item()}")

        # 3. 彻底删除区域外的高斯
        if outside_mask.sum() > 0:
            print(f"   🗑️  删除区域外的 {outside_mask.sum().item()} 个高斯...")
            self.prune_anchor(outside_mask)
            # 更新 inside mask (删除后mask长度变化)
            self._current_region_inside_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device='cuda')
            print(f"   ✅ 清理完成，剩余高斯数量: {self._anchor.shape[0]}")
        else:
            print(f"   ✅ 无区域外高斯需要删除")
    def reset_from_ply(self, ply_path: str):
        """
        【增强】从PLY加载预训练高斯 + 自动冻结所有高斯参数
        """
        print(f"\n==================================================")
        print(f"🔄 【冻结模式】从预训练PLY加载高斯，冻结所有高斯球参数")
        print(f"📂 加载路径：{ply_path}")
        
        # 1. 清空原有高斯
        if self._anchor.shape[0] > 0:
            delete_all_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
            self.prune_anchor(delete_all_mask)

        # 2. 加载PLY高斯
        self.load_ply_sparse_gaussian(ply_path)
        print(f"✅ 加载完成，高斯数量：{self._anchor.shape[0]}")

        # 3. ====================== 核心：冻结所有高斯参数 ======================
        self._anchor.requires_grad_(False)
        self._offset.requires_grad_(False)
        self._anchor_feat.requires_grad_(False)
        self._opacity.requires_grad_(False)
        self._scaling.requires_grad_(False)
        self._rotation.requires_grad_(False)
        print(f"✅ 已冻结：位置/偏移/特征/不透明度/缩放/旋转")

        # 4. 初始化统计量（冻结模式下置空，不使用）
        self.opacity_accum = torch.empty(0, device="cuda")
        self.offset_gradient_accum = torch.empty(0, device="cuda")
        self.offset_denom = torch.empty(0, device="cuda")
        self.anchor_demon = torch.empty(0, device="cuda")
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")

        # 5. 重置优化器（仅保留MLP参数）
        if self.optimizer is not None:
            self.optimizer.state.clear()
            # self.training_setup(self.training_args)  

        print(f"==================================================\n")

