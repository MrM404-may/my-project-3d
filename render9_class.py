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

class Render9Huang:
    def __init__(self, model_path, source_path="", iteration=-1, resolution_scales=None, batch_size=1657, quiet=True):
        """
        初始化Render9Huang类
        
        Args:
            model_path: 模型路径
            source_path: 源数据路径
            iteration: 要加载的迭代次数，默认为-1（最新）
            resolution_scales: 分辨率缩放列表
            batch_size: 批量加载相机的大小
            quiet: 是否安静模式
        """
        self.model_path = model_path
        self.source_path = source_path
        self.iteration = iteration
        self.resolution_scales = resolution_scales if resolution_scales else [1.0]
        self.batch_size = batch_size
        self.quiet = quiet
        
        # 初始化参数
        parser = ArgumentParser(description="Render9Huang parameters")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        
        # 创建默认参数
        class Args:
            def __init__(self):
                self.model_path = model_path
                self.source_path = source_path
                self.iteration = iteration
                self.skip_train = False
                self.skip_test = False
                self.quiet = quiet
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
            self.gaussians.plot_levels()
            
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
        
    def load_camera_region_mapping(self, regions_config_path=None, cache_path=None):
        """
        加载相机ID到区域的映射
        
        Args:
            regions_config_path: 区域配置文件路径
            cache_path: 相机信息缓存文件路径
        
        Returns:
            camera_id_to_region: 相机ID到区域索引的映射
        """
        camera_id_to_region = {}
        
        # 默认路径
        if regions_config_path is None:
            if self.source_path:
                regions_config_path = os.path.join(self.source_path, "regions_config.json")
            else:
                regions_config_path = f"/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma3w/regions_config.json"
        
        if cache_path is None:
            if self.source_path:
                cache_path = os.path.join(self.source_path, "cache.json")
            else:
                cache_path = f"/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma3w/cache.json"
        
        if os.path.exists(regions_config_path):
            with open(regions_config_path, 'r', encoding='utf-8') as f:
                regions_config = json.load(f)
            
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    camera_info_dict = json.load(f)
                
                print("===== 开始匹配相机与区域 =====")
                
                # 核心匹配逻辑
                for cam_id, cam_data in camera_info_dict.items():
                    if cam_id not in camera_id_to_region:  # 确保相机只被分配一次
                        for region_idx, region in enumerate(regions_config):
                            region_name = region['name']
                            if cam_data['in_regions'].get(region_name, False) and (int(cam_id)<int(region['max_id'])):
                                camera_id_to_region[cam_id] = region_idx
                                break   
                
                # 统计各区域详情
                print("\n===== 各区域详细信息统计表 =====")
                
                # 初始化：每个区域存 [相机数量, 实际最大camid]
                region_stats = {i: [0, -1] for i in range(len(regions_config))}
                
                # 正确遍历：相机ID → 对应区域
                for cam_id, region_idx in camera_id_to_region.items():
                    cam = int(cam_id)
                    region_stats[region_idx][0] += 1  # 计数+1
                    if cam > region_stats[region_idx][1]:
                        region_stats[region_idx][1] = cam  # 只更新当前区域自己的最大ID
                
                # 打印表头
                print(f"{'区域名称':<12} | {'索引':<4} | {'实际最大CamID':<12} | {'匹配相机数':<8}")
                print("-" * 70)
                
                # 输出结果
                for region_idx, region in enumerate(regions_config):
                    count, real_max = region_stats[region_idx]
                    if real_max == -1:
                        real_max = "无"
                    print(f"{region['name']:<12} | {region_idx:<4} | {real_max:<12} | {count:<8}")
            else:
                print(f"错误：缓存文件不存在！路径：{cache_path}")
        else:
            print(f"错误：区域配置文件不存在！路径：{regions_config_path}")
        
        return camera_id_to_region
    
    def render_set(self, name, iteration, views, gaussians, pipeline, background, show_level, ape_code):
        """
        渲染一组相机
        
        Args:
            name: 数据集名称（train/test）
            iteration: 迭代次数
            views: 相机列表
            gaussians: 高斯模型
            pipeline: 渲染管道参数
            background: 背景颜色
            show_level: 是否显示不同级别的渲染
            ape_code: 外观编码
        """
        render_path = os.path.join(self.model_path, name, "ours_{}".format(iteration), "renders")
        makedirs(render_path, exist_ok=True)
        gts_path = os.path.join(self.model_path, name, "ours_{}".format(iteration), "gt")

        inv_depth_path = os.path.join(self.model_path, name, "ours_{}".format(iteration), "inv_depth")
        makedirs(inv_depth_path, exist_ok=True)

        makedirs(gts_path, exist_ok=True)
        if show_level:
            render_level_path = os.path.join(self.model_path, name, "ours_{}".format(iteration), "renders_level")
            makedirs(render_level_path, exist_ok=True)

        # 加载区域配置和相机ID到区域的映射
        camera_id_to_region = self.load_camera_region_mapping()

        t_list = []
        per_view_dict = {}
        per_view_level_dict = {}

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            camera_region = camera_id_to_region.get(str(getattr(view, 'uid', '')), 0)  # 安全获取区域ID，默认0
            if hasattr(camera_region, '__len__') and len(camera_region) > 2:
                print(f"Warning: Camera {view.uid} has multiple region matches: {camera_region}. Using the first match.")
            torch.cuda.synchronize(); t0 = time.time()

            gaussians.set_anchor_mask(view.camera_center, iteration, view.resolution_scale)
            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
            render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, ape_code=idx, camera_region=camera_region)
            
            torch.cuda.synchronize(); t1 = time.time()
            t_list.append(t1-t0)

            rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
            visible_count = render_pkg["visibility_filter"].sum()  
            per_view_dict['{0:05d}'.format(idx)+".png"] = visible_count.item()

            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

            inv_depth_map = render_pkg["inv_depth"].detach().cpu().numpy()
            np.save(os.path.join(inv_depth_path, f"{idx}_inv_depth.npy"), inv_depth_map)
            if show_level:
                for cur_level in range(gaussians.levels):
                    gaussians.set_anchor_mask_perlevel(view.camera_center, view.resolution_scale, cur_level)
                    voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
                    render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, ape_code=ape_code, camera_region=camera_region)
                    
                    rendering = render_pkg["render"]
                    visible_count = render_pkg["visibility_filter"].sum()
                    
                    torchvision.utils.save_image(rendering, os.path.join(render_level_path, '{0:05d}_LOD{1:d}'.format(idx, cur_level) + ".png"))
                    per_view_level_dict['{0:05d}_LOD{1:d}'.format(idx, cur_level) + ".png"] = visible_count.item()

        t = np.array(t_list[5:])
        fps = 1.0 / t.mean()
        print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

        with open(os.path.join(self.model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True) 
        if show_level:
            with open(os.path.join(self.model_path, name, "ours_{}".format(iteration), "per_view_count_level.json"), 'w') as fp:
                json.dump(per_view_level_dict, fp, indent=True)     
    
    def render_sets(self, skip_train=False, skip_test=False, show_level=False, ape_code=10):
        """
        渲染训练集和测试集
        
        Args:
            skip_train: 是否跳过训练集
            skip_test: 是否跳过测试集
            show_level: 是否显示不同级别的渲染
            ape_code: 外观编码
        """
        if self.gaussians is None or self.scene is None:
            raise ValueError("Scene and gaussians not loaded. Call load_scene_and_gaussians() first.")
        
        with torch.no_grad():
            if not skip_train:
                # 使用批量加载模式，每次只渲染一个批次的相机
                num_batches = self.scene.get_num_batches()
                for batch_idx in range(num_batches):
                    batch_cameras = self.scene.getTrainCameras(batch_idx)
                    if not batch_cameras:
                        continue
                    print(f"Rendering train set, batch {batch_idx+1}/{num_batches}...")
                    self.render_set(self.model_params.model_path, "train", self.scene.loaded_iter, batch_cameras, 
                                   self.gaussians, self.pipeline_params, self.background, show_level, batch_idx)
                    # 释放内存
                    del batch_cameras
                    torch.cuda.empty_cache()

            if not skip_test:
                self.render_set(self.model_params.model_path, "test", self.scene.loaded_iter, self.scene.getTestCameras(), 
                               self.gaussians, self.pipeline_params, self.background, show_level, ape_code)

def main():
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Render9Huang testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=10, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show_level", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # 初始化系统状态 (RNG)
    safe_state(args.quiet)

    # 创建Render9Huang实例
    renderer = Render9Huang(
        model_path=args.model_path,
        source_path=getattr(args, 'source_path', ''),
        iteration=args.iteration,
        resolution_scales=args.resolution_scales,
        quiet=args.quiet
    )
    
    # 加载场景和高斯模型
    renderer.load_scene_and_gaussians()
    
    # 渲染训练集和测试集
    renderer.render_sets(
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        show_level=args.show_level,
        ape_code=args.ape
    )

if __name__ == "__main__":
    main()
