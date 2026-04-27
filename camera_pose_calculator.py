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
import os
import torch
import numpy as np
from scene import Scene
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from arguments import ModelParams, get_combined_args
from argparse import ArgumentParser
import json

class CameraPoseCalculator:
    def __init__(self, model_path, iteration=-1, regions_config_path=None, cache_path=None):
        """
        初始化相机位姿计算器
        
        Args:
            model_path: 模型路径
            iteration: 要加载的迭代次数，默认为-1（最新）
            regions_config_path: 区域配置文件路径
            cache_path: 相机信息缓存文件路径
        """
        self.model_path = model_path
        self.iteration = iteration
        self.regions_config_path = regions_config_path
        self.cache_path = cache_path
        
        # 加载场景以获取相机参数
        self.load_scene_params()
        
        # 加载区域配置和相机ID到区域的映射
        self.regions_config = None
        self.camera_id_to_region = {}
        self.load_regions_config()
        self.load_camera_region_mapping()
    
    def load_scene_params(self):
        """
        加载场景参数，获取相机的基本信息
        """
        # 初始化参数
        parser = ArgumentParser(description="Camera pose calculator parameters")
        model = ModelParams(parser, sentinel=True)
        
        # 创建默认参数
        class Args:
            def __init__(self):
                self.model_path = self.model_path
                self.iteration = self.iteration
                self.quiet = True
        
        # 这里需要使用正确的作用域
        args = Args()
        args.model_path = self.model_path
        args.iteration = self.iteration
        
        self.model_params = model.extract(args)
        
        # 加载场景以获取相机信息
        with torch.no_grad():
            self.gaussians = GaussianModel(
                self.model_params.feat_dim, self.model_params.n_offsets, self.model_params.fork, 
                self.model_params.use_feat_bank, self.model_params.appearance_dim, 
                self.model_params.add_opacity_dist, self.model_params.add_cov_dist, 
                self.model_params.add_color_dist, self.model_params.add_level, 
                self.model_params.visible_threshold, self.model_params.dist2level, 
                self.model_params.base_layer, self.model_params.progressive, self.model_params.extend
            )
            
            # 加载场景
            self.scene = Scene(
                self.model_params, self.gaussians, 
                load_iteration=self.iteration, 
                shuffle=False, 
                resolution_scales=[1.0], 
                batch_size=100
            )
            
            # 获取第一个相机的参数作为参考
            cameras = self.scene.getTrainCameras()
            if not cameras:
                cameras = self.scene.getTestCameras()
            
            if cameras:
                self.reference_camera = cameras[0]
                print(f"Loaded reference camera with FoVx: {self.reference_camera.FoVx}, FoVy: {self.reference_camera.FoVy}")
                print(f"Image size: {self.reference_camera.image_width}x{self.reference_camera.image_height}")
            else:
                raise ValueError("No cameras found in the scene")
    
    def load_regions_config(self):
        """
        加载区域配置文件
        """
        if self.regions_config_path and os.path.exists(self.regions_config_path):
            with open(self.regions_config_path, 'r', encoding='utf-8') as f:
                self.regions_config = json.load(f)
            print(f"Loaded regions config with {len(self.regions_config)} regions")
        else:
            print("No regions config file provided or found")
    
    def load_camera_region_mapping(self):
        """
        加载相机ID到区域的映射
        """
        if self.cache_path and os.path.exists(self.cache_path) and self.regions_config:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                camera_info_dict = json.load(f)
            
            # 构建相机ID到区域的映射
            for cam_id, cam_data in camera_info_dict.items():
                if cam_id not in self.camera_id_to_region:
                    for region_idx, region in enumerate(self.regions_config):
                        region_name = region['name']
                        if cam_data['in_regions'].get(region_name, False) and (int(cam_id)<int(region['max_id'])):
                            self.camera_id_to_region[cam_id] = region_idx
                            break
            
            print(f"Loaded camera to region mapping for {len(self.camera_id_to_region)} cameras")
        else:
            print("No cache file provided or found, or no regions config loaded")
    
    def calculate_camera_similarity(self, new_camera, existing_cameras):
        """
        计算新相机与现有相机的相似性
        
        Args:
            new_camera: 新相机对象
            existing_cameras: 现有相机列表
            
        Returns:
            最相似的相机及其相似度
        """
        if not existing_cameras:
            return None, 0.0
        
        max_similarity = -1
        most_similar_camera = None
        
        # 计算新相机的位置和方向
        new_pos = new_camera.camera_center
        
        # 从world_view_transform提取旋转矩阵
        new_rot = new_camera.world_view_transform[:3, :3]
        
        for existing_camera in existing_cameras:
            # 计算位置相似度（距离的倒数）
            existing_pos = existing_camera.camera_center
            pos_dist = torch.norm(new_pos - existing_pos).item()
            pos_similarity = 1.0 / (1.0 + pos_dist)  # 距离越近，相似度越高
            
            # 计算旋转相似度（旋转矩阵的点积）
            existing_rot = existing_camera.world_view_transform[:3, :3]
            rot_similarity = torch.trace(torch.mm(new_rot, existing_rot.t())).item() / 3.0
            rot_similarity = (rot_similarity + 1.0) / 2.0  # 归一化到[0,1]
            
            # 综合相似度
            total_similarity = 0.7 * pos_similarity + 0.3 * rot_similarity
            
            if total_similarity > max_similarity:
                max_similarity = total_similarity
                most_similar_camera = existing_camera
        
        return most_similar_camera, max_similarity
    
    def assign_camera_region(self, new_camera):
        """
        为新相机分配区域
        
        Args:
            new_camera: 新相机对象
            
        Returns:
            分配的区域索引和对应的ape_code
        """
        # 获取所有现有相机
        all_cameras = self.scene.getTrainCameras() + self.scene.getTestCameras()
        
        # 找到最相似的相机
        most_similar_camera, similarity = self.calculate_camera_similarity(new_camera, all_cameras)
        
        if most_similar_camera:
            # 获取最相似相机的区域
            cam_id = str(getattr(most_similar_camera, 'uid', ''))
            region_idx = self.camera_id_to_region.get(cam_id, 0)  # 默认区域0
            
            # 使用相似相机的uid作为ape_code
            ape_code = getattr(most_similar_camera, 'uid', 0)
            
            print(f"Assigned region {region_idx} to new camera (similarity: {similarity:.4f})")
            print(f"Using ape_code: {ape_code}")
            
            return region_idx, ape_code
        else:
            # 如果没有相似相机，使用默认区域和ape_code
            print("No similar cameras found, using default region 0")
            return 0, 0
    
    def calculate_camera_transforms(self, R, T, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        """
        计算新相机的变换矩阵和相机中心
        
        Args:
            R: 3x3旋转矩阵
            T: 3x1平移向量
            trans: 额外的平移向量
            scale: 缩放因子
            
        Returns:
            camera_center: 相机中心
            world_view_transform: 世界到相机的变换矩阵
            full_proj_transform: 完整的投影变换矩阵
        """
        # 计算世界到相机的变换矩阵
        world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        
        # 计算投影矩阵
        projection_matrix = getProjectionMatrix(
            znear=self.reference_camera.znear, 
            zfar=self.reference_camera.zfar, 
            fovX=self.reference_camera.FoVx, 
            fovY=self.reference_camera.FoVy
        ).transpose(0, 1).cuda()
        
        # 计算完整的投影变换矩阵
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        
        # 计算相机中心
        camera_center = world_view_transform.inverse()[3, :3]
        
        return camera_center, world_view_transform, full_proj_transform
    
    def create_camera_from_pose(self, R, T, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        """
        根据位姿创建相机对象
        
        Args:
            R: 3x3旋转矩阵
            T: 3x1平移向量
            trans: 额外的平移向量
            scale: 缩放因子
            
        Returns:
            camera: MiniCam对象
        """
        # 计算变换矩阵
        _, world_view_transform, full_proj_transform = self.calculate_camera_transforms(R, T, trans, scale)
        
        # 创建MiniCam对象
        from scene.cameras import MiniCam
        camera = MiniCam(
            width=self.reference_camera.image_width,
            height=self.reference_camera.image_height,
            fovy=self.reference_camera.FoVy,
            fovx=self.reference_camera.FoVx,
            znear=self.reference_camera.znear,
            zfar=self.reference_camera.zfar,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform
        )
        
        return camera

def main():
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Camera pose calculator")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--regions_config", default=None, type=str, help="Path to regions config file")
    parser.add_argument("--cache", default=None, type=str, help="Path to camera info cache file")
    args = get_combined_args(parser)
    
    print(f"Loading scene from {args.model_path}")
    
    # 创建计算器
    calculator = CameraPoseCalculator(
        model_path=args.model_path, 
        iteration=args.iteration,
        regions_config_path=args.regions_config,
        cache_path=args.cache
    )
    
    # 示例：输入新相机的位姿
    # 这里使用单位矩阵作为旋转矩阵，零向量作为平移向量
    # 实际使用时，用户需要输入真实的位姿信息
    R = np.eye(3)  # 单位旋转矩阵
    T = np.array([0.0, 0.0, 0.0])  # 零平移向量
    
    print("\nCalculating camera transforms for new pose...")
    print(f"Rotation matrix:\n{R}")
    print(f"Translation vector: {T}")
    
    # 计算变换矩阵
    camera_center, world_view_transform, full_proj_transform = calculator.calculate_camera_transforms(R, T)
    
    print("\nCalculation results:")
    print(f"Camera center: {camera_center.cpu().numpy()}")
    print(f"World view transform shape: {world_view_transform.shape}")
    print(f"Full proj transform shape: {full_proj_transform.shape}")
    
    # 创建相机对象
    camera = calculator.create_camera_from_pose(R, T)
    print(f"\nCreated camera with center: {camera.camera_center.cpu().numpy()}")
    
    # 分配区域和计算ape_code
    region_idx, ape_code = calculator.assign_camera_region(camera)
    print(f"\nAssigned region: {region_idx}")
    print(f"Calculated ape_code: {ape_code}")
    
    # 演示如何使用这些值进行渲染
    print("\nTo use this camera in rendering, you can pass the following parameters:")
    print(f"- camera_region: {region_idx}")
    print(f"- ape_code: {ape_code}")

if __name__ == "__main__":
    main()
