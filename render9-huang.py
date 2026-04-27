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


class Renderer:
    def __init__(self, regions_config_path=None, cache_path=None):
        self.regions_config_path = regions_config_path or "/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma0422/regions_config.json"
        self.cache_path = cache_path or "/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma0422/cache.json"
        self.camera_id_to_region = {}
        self.regions_config = []
    
    def load_region_config(self):
        camera_id_to_region = {}
        if os.path.exists(self.regions_config_path):
            with open(self.regions_config_path, 'r', encoding='utf-8') as f:
                self.regions_config = json.load(f)
            
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    camera_info_dict = json.load(f)

                print("===== 开始匹配相机与区域 =====")
                for cam_id, cam_data in camera_info_dict.items():
                    if cam_id not in camera_id_to_region:
                        for region_idx, region in enumerate(self.regions_config):
                            region_name = region['name']
                            if cam_data['in_regions'].get(region_name, False) and (int(cam_id)<int(region['max_id'])):
                                camera_id_to_region[cam_id] = region_idx
                                break   
                
                print("\n===== 各区域详细信息统计表 =====")
                region_stats = {i: [0, -1] for i in range(len(self.regions_config))}

                for cam_id, region_idx in camera_id_to_region.items():
                    cam = int(cam_id)
                    region_stats[region_idx][0] += 1
                    if cam > region_stats[region_idx][1]:
                        region_stats[region_idx][1] = cam

                print(f"{'区域名称':<12} | {'索引':<4} | {'实际最大CamID':<12} | {'匹配相机数':<8}")
                print("-" * 70)

                for region_idx, region in enumerate(self.regions_config):
                    count, real_max = region_stats[region_idx]
                    if real_max == -1:
                        real_max = "无"
                    print(f"{region['name']:<12} | {region_idx:<4} | {real_max:<12} | {count:<8}")
            else:
                print(f"错误：缓存文件不存在！路径：{self.cache_path}")
        else:
            print(f"错误：区域配置文件不存在！路径：{self.regions_config_path}")
        
        self.camera_id_to_region = camera_id_to_region
        return camera_id_to_region
    
    def render_set(self, model_path, name, iteration, views, gaussians, pipeline, background, show_level, ape_code):
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        makedirs(render_path, exist_ok=True)
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

        inv_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "inv_depth")
        makedirs(inv_depth_path, exist_ok=True)

        makedirs(gts_path, exist_ok=True)
        if show_level:
            render_level_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_level")
            makedirs(render_level_path, exist_ok=True)

        if not self.camera_id_to_region:
            self.load_region_config()

        t_list = []
        per_view_dict = {}
        per_view_level_dict = {}
        all_cameras_info = []
        camera_region = 0
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            camera_region = self.camera_id_to_region.get(str(getattr(view, 'uid', '')), 0)
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
            region_save_path = os.path.join(render_path, str(camera_region))
            os.makedirs(region_save_path, exist_ok=True)

            torchvision.utils.save_image(
                rendering, 
                os.path.join(region_save_path, f'{idx}.png')
            )
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

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

        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True) 
        if show_level:
            with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count_level.json"), 'w') as fp:
                json.dump(per_view_level_dict, fp, indent=True)     
    
    def render_set_one_view(self, dataset : ModelParams, iteration : int, pipeline : PipelineParams, R, T, show_level : bool, ape_code : int = 10):
        with torch.no_grad():
            gaussians = GaussianModel(
                dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
                dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
                dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
            )
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=dataset.resolution_scales)
            gaussians.eval()
            gaussians.plot_levels()
            
            if dataset.random_background:
                bg_color = [np.random.random(),np.random.random(),np.random.random()] 
            elif dataset.white_background:
                bg_color = [1.0, 1.0, 1.0]
            else:
                bg_color = [0.0, 0.0, 0.0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            if not os.path.exists(dataset.model_path):
                os.makedirs(dataset.model_path)
            
            # 获取一个测试相机作为模板
            test_cameras = scene.getTestCameras()
            if not test_cameras:
                print("Error: No test cameras available")
                return
            
            # 使用第一个相机作为模板
            view = test_cameras[0]
            # 设置相机的旋转矩阵和平移向量
            view.R = R
            view.T = T
            
            # 渲染单个视图
            self.render_set(dataset.model_path, "test", scene.loaded_iter, [view], gaussians, pipeline, background, show_level, ape_code)
    
    def render_sets(self, dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, show_level : bool, ape_code : int):
        with torch.no_grad():
            gaussians = GaussianModel(
                dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
                dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
                dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
            )
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=dataset.resolution_scales)
            gaussians.eval()
            gaussians.plot_levels()
            if dataset.random_background:
                bg_color = [np.random.random(),np.random.random(),np.random.random()] 
            elif dataset.white_background:
                bg_color = [1.0, 1.0, 1.0]
            else:
                bg_color = [0.0, 0.0, 0.0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            if not os.path.exists(dataset.model_path):
                os.makedirs(dataset.model_path)
            ape_code  = 10
            if not skip_train:
                num_batches = 588
                for batch_idx in range(num_batches):
                    batch_cameras = scene.get_unloaded_cameras(range(num_batches))
                    if not batch_cameras:
                        continue
                    print(f"Rendering train set, batch {batch_idx+1}/{num_batches}...")
                    self.render_set(dataset.model_path, "train", scene.loaded_iter, batch_cameras, gaussians, pipeline, background, show_level, batch_idx)
                    del batch_cameras
                    torch.cuda.empty_cache()

            if not skip_test:
                self.render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, show_level, ape_code)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, show_level, ape_code):
    renderer = Renderer()
    renderer.render_set(model_path, name, iteration, views, gaussians, pipeline, background, show_level, ape_code)


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, show_level : bool, ape_code : int):
    renderer = Renderer()
    renderer.render_sets(dataset, iteration, pipeline, skip_train, skip_test, show_level, ape_code)


def render_set_one_view(dataset : ModelParams, iteration : int, pipeline : PipelineParams, R, T, show_level : bool, ape_code : int = 10):
    renderer = Renderer()
    renderer.render_set_one_view(dataset, iteration, pipeline, R, T, show_level, ape_code)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
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

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.show_level, args.ape)
    
