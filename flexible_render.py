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
from os import makedirs
import torch
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

class FlexibleRenderer:
    def __init__(self, model_path, iteration=-1, resolution_scales=None, batch_size=100):
        """
        初始化灵活渲染器
        
        Args:
            model_path: 模型路径
            iteration: 要加载的迭代次数，默认为-1（最新）
            resolution_scales: 分辨率缩放列表
            batch_size: 批量加载相机的大小
        """
        self.model_path = model_path
        self.iteration = iteration
        self.resolution_scales = resolution_scales if resolution_scales else [1.0]
        self.batch_size = batch_size
        
        # 初始化参数
        parser = ArgumentParser(description="Flexible rendering script parameters")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        
        # 创建默认参数
        class Args:
            def __init__(self):
                self.model_path = model_path
                self.iteration = iteration
                self.skip_train = False
                self.skip_test = False
                self.quiet = True
                self.show_level = False
                self.white_background = False
                self.random_background = False
                self.resolution_scales = resolution_scales
        
        args = Args()
        self.model_params = model.extract(args)
        self.pipeline_params = pipeline.extract(args)
        
        # 加载场景和高斯模型
        self.gaussians = None
        self.scene = None
        self.background = None
        
    def load_scene_and_gaussians(self):
        """
        加载场景和高斯模型
        """
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
                resolution_scales=self.resolution_scales, 
                batch_size=self.batch_size
            )
            
            self.gaussians.eval()
            print(f"Loaded scene with {len(self.scene.getTrainCameras())} train cameras and {len(self.scene.getTestCameras())} test cameras")
            
            # 设置背景
            if self.model_params.random_background:
                self.background = torch.tensor([np.random.random(), np.random.random(), np.random.random()], 
                                             dtype=torch.float32, device="cuda")
            elif self.model_params.white_background:
                self.background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
            else:
                self.background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
            
            # 确保输出目录存在
            if not os.path.exists(self.model_params.model_path):
                os.makedirs(self.model_params.model_path)
        
    def render_camera(self, camera, output_path, show_level=False, ape_code=-1, camera_region=0):
        """
        渲染单个相机
        
        Args:
            camera: 相机对象
            output_path: 输出路径
            show_level: 是否显示不同级别的渲染
            ape_code: 外观编码
            camera_region: 相机区域
        """
        if self.gaussians is None or self.scene is None:
            raise ValueError("Scene and gaussians not loaded. Call load_scene_and_gaussians() first.")
        
        # 创建输出目录
        render_path = os.path.join(output_path, "renders")
        inv_depth_path = os.path.join(output_path, "inv_depth")
        makedirs(render_path, exist_ok=True)
        makedirs(inv_depth_path, exist_ok=True)
        
        if show_level:
            render_level_path = os.path.join(output_path, "renders_level")
            makedirs(render_level_path, exist_ok=True)
        
        # 渲染
        torch.cuda.synchronize()
        t0 = time.time()
        
        self.gaussians.set_anchor_mask(camera.camera_center, self.scene.loaded_iter, camera.resolution_scale)
        voxel_visible_mask = prefilter_voxel(camera, self.gaussians, self.pipeline_params, self.background)
        render_pkg = render(camera, self.gaussians, self.pipeline_params, self.background, 
                          visible_mask=voxel_visible_mask, ape_code=ape_code, camera_region=camera_region)
        
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"Rendering time: {t1 - t0:.4f} seconds")
        
        # 保存渲染结果
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"camera_{camera.uid}.png"))
        
        # 保存逆深度图
        inv_depth_map = render_pkg["inv_depth"].detach().cpu().numpy()
        np.save(os.path.join(inv_depth_path, f"camera_{camera.uid}_inv_depth.npy"), inv_depth_map)
        
        # 保存不同级别的渲染（如果需要）
        if show_level:
            for cur_level in range(self.gaussians.levels):
                self.gaussians.set_anchor_mask_perlevel(camera.camera_center, camera.resolution_scale, cur_level)
                voxel_visible_mask = prefilter_voxel(camera, self.gaussians, self.pipeline_params, self.background)
                render_pkg = render(camera, self.gaussians, self.pipeline_params, self.background, 
                                  visible_mask=voxel_visible_mask, ape_code=ape_code, camera_region=camera_region)
                
                rendering = render_pkg["render"]
                torchvision.utils.save_image(rendering, os.path.join(render_level_path, f"camera_{camera.uid}_LOD{cur_level}.png"))
        
        return render_pkg
    
    def render_cameras(self, cameras, output_path, show_level=False, ape_code=-1):
        """
        渲染多个相机
        
        Args:
            cameras: 相机列表
            output_path: 输出路径
            show_level: 是否显示不同级别的渲染
            ape_code: 外观编码
        """
        if self.gaussians is None or self.scene is None:
            raise ValueError("Scene and gaussians not loaded. Call load_scene_and_gaussians() first.")
        
        # 创建输出目录
        render_path = os.path.join(output_path, "renders")
        inv_depth_path = os.path.join(output_path, "inv_depth")
        makedirs(render_path, exist_ok=True)
        makedirs(inv_depth_path, exist_ok=True)
        
        if show_level:
            render_level_path = os.path.join(output_path, "renders_level")
            makedirs(render_level_path, exist_ok=True)
        
        t_list = []
        per_view_dict = {}
        
        for idx, camera in enumerate(tqdm(cameras, desc="Rendering progress")):
            torch.cuda.synchronize()
            t0 = time.time()
            
            self.gaussians.set_anchor_mask(camera.camera_center, self.scene.loaded_iter, camera.resolution_scale)
            voxel_visible_mask = prefilter_voxel(camera, self.gaussians, self.pipeline_params, self.background)
            render_pkg = render(camera, self.gaussians, self.pipeline_params, self.background, 
                              visible_mask=voxel_visible_mask, ape_code=ape_code)
            
            torch.cuda.synchronize()
            t1 = time.time()
            t_list.append(t1 - t0)
            
            # 保存渲染结果
            rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
            visible_count = render_pkg["visibility_filter"].sum()
            per_view_dict[f'{idx:05d}.png'] = visible_count.item()
            
            torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
            
            # 保存逆深度图
            inv_depth_map = render_pkg["inv_depth"].detach().cpu().numpy()
            np.save(os.path.join(inv_depth_path, f"{idx}_inv_depth.npy"), inv_depth_map)
            
            # 保存不同级别的渲染（如果需要）
            if show_level:
                for cur_level in range(self.gaussians.levels):
                    self.gaussians.set_anchor_mask_perlevel(camera.camera_center, camera.resolution_scale, cur_level)
                    voxel_visible_mask = prefilter_voxel(camera, self.gaussians, self.pipeline_params, self.background)
                    render_pkg = render(camera, self.gaussians, self.pipeline_params, self.background, 
                                      visible_mask=voxel_visible_mask, ape_code=ape_code)
                    
                    rendering = render_pkg["render"]
                    torchvision.utils.save_image(rendering, os.path.join(render_level_path, f'{idx:05d}_LOD{cur_level}.png'))
        
        # 计算平均FPS
        if t_list:
            t = np.array(t_list)
            fps = 1.0 / t.mean()
            print(f'Average FPS: {fps:.5f}')
        
        # 保存可见性统计
        with open(os.path.join(output_path, "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    def render_train_set(self, output_path, show_level=False):
        """
        渲染训练集
        
        Args:
            output_path: 输出路径
            show_level: 是否显示不同级别的渲染
        """
        if self.gaussians is None or self.scene is None:
            raise ValueError("Scene and gaussians not loaded. Call load_scene_and_gaussians() first.")
        
        # 使用批量加载模式
        num_batches = self.scene.get_num_batches()
        for batch_idx in range(num_batches):
            batch_cameras = self.scene.getTrainCameras(batch_idx)
            if not batch_cameras:
                continue
            
            batch_output_path = os.path.join(output_path, f"batch_{batch_idx}")
            print(f"Rendering train batch {batch_idx+1}/{num_batches}...")
            self.render_cameras(batch_cameras, batch_output_path, show_level, batch_idx)
            
            # 释放内存
            del batch_cameras
            torch.cuda.empty_cache()
    
    def render_test_set(self, output_path, show_level=False, ape_code=10):
        """
        渲染测试集
        
        Args:
            output_path: 输出路径
            show_level: 是否显示不同级别的渲染
            ape_code: 外观编码
        """
        if self.gaussians is None or self.scene is None:
            raise ValueError("Scene and gaussians not loaded. Call load_scene_and_gaussians() first.")
        
        test_cameras = self.scene.getTestCameras()
        if test_cameras:
            print(f"Rendering test set with {len(test_cameras)} cameras...")
            self.render_cameras(test_cameras, output_path, show_level, ape_code)
        else:
            print("No test cameras found.")

def main():
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Flexible rendering script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=10, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show_level", action="store_true")
    parser.add_argument("--output_path", default=None, type=str, help="Custom output path")
    args = get_combined_args(parser)
    
    print(f"Rendering {args.model_path}")
    
    # 初始化系统状态
    safe_state(args.quiet)
    
    # 创建渲染器
    renderer = FlexibleRenderer(
        model_path=args.model_path,
        iteration=args.iteration,
        resolution_scales=args.resolution_scales
    )
    
    # 加载场景和高斯模型
    renderer.load_scene_and_gaussians()
    
    # 确定输出路径
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = os.path.join(args.model_path, "flexible_renders")
    
    # 渲染训练集
    if not args.skip_train:
        train_output_path = os.path.join(output_path, "train")
        renderer.render_train_set(train_output_path, args.show_level)
    
    # 渲染测试集
    if not args.skip_test:
        test_output_path = os.path.join(output_path, "test")
        renderer.render_test_set(test_output_path, args.show_level, args.ape)

if __name__ == "__main__":
    main()
