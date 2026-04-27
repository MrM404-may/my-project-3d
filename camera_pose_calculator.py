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

class CameraPoseCalculator:
    def __init__(self, model_path, iteration=-1):
        """
        初始化相机位姿计算器
        
        Args:
            model_path: 模型路径
            iteration: 要加载的迭代次数，默认为-1（最新）
        """
        self.model_path = model_path
        self.iteration = iteration
        
        # 加载场景以获取相机参数
        self.load_scene_params()
    
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
    args = get_combined_args(parser)
    
    print(f"Loading scene from {args.model_path}")
    
    # 创建计算器
    calculator = CameraPoseCalculator(args.model_path, args.iteration)
    
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

if __name__ == "__main__":
    main()
