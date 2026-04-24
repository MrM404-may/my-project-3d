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
from read_mapping import get_cameras_json_id_by_mulp_id, get_full_mapping_by_mulp_id
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, show_level, ape_code):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    inv_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "inv_depth")
    makedirs(inv_depth_path, exist_ok=True)

    makedirs(gts_path, exist_ok=True)
    if show_level:
        render_level_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_level")
        makedirs(render_level_path, exist_ok=True)

    # 加载区域配置和相机ID到区域的映射
    regions_config_path = f"/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma3w/regions_config.json"
    camera_id_to_region = {}
    if os.path.exists(regions_config_path):
        with open(regions_config_path, 'r', encoding='utf-8') as f:
            regions_config = json.load(f)
        
        # 加载相机ID到区域的映射
        cache_path = f"/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma3w/cache.json"
        camera_id_to_region = {}
        if os.path.exists(cache_path):
            # 修复缩进：with 必须缩进在 if 内部
            with open(cache_path, 'r', encoding='utf-8') as f:
                camera_info_dict = json.load(f)

            # ===================== 核心匹配逻辑 =====================
            print("===== 开始匹配相机与区域 =====")
            # for cam_id, cam_data in camera_info_dict.items():
            #     # 遍历所有区域配置
            #     for region_idx, region in enumerate(regions_config):
            #         region_name = region['name']
            #         # 安全获取：相机是否在当前区域内
            #         in_region = cam_data['in_regions'].get(region_name, False)
            #         if in_region and (int(cam_id)<int(region['max_id'])):
            #             camera_id_to_region[cam_id] = region_idx
            #             # 调试打印：清晰看到匹配结果
            #             print(f"相机ID: {cam_id} -> 匹配区域: {region_name} (索引: {region_idx})")
            #             break  # 找到第一个匹配的区域就退出循环
            for cam_id, cam_data in camera_info_dict.items():
                if cam_id not in camera_id_to_region:  # 确保相机只被分配一次
                    for region_idx, region in enumerate(regions_config):
                        region_name = region['name']
                        if cam_data['in_regions'].get(region_name, False) and (int(cam_id)<int(region['max_id'])):
                            camera_id_to_region[cam_id] = region_idx
                            break   
             # ===================== 统计1：各区域详情（含区域自身Max ID） =====================
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

    t_list = []
    per_view_dict = {}
    per_view_level_dict = {}
    # 存储所有相机逆深度信息
    all_cameras_info = []
    # 获取相机的区域ID
    camera_region = 0  # 默认区域0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):


        uid_temp = get_cameras_json_id_by_mulp_id(int(view.uid)+1)
        
        camera_region = camera_id_to_region.get(str(uid_temp), 0)  # 安全获取区域ID，默认0
        print(f"\n正在渲染相机 {view.uid} (Cameras.json ID: {uid_temp}, 区域ID: {camera_region})...")
        if hasattr(camera_region, '__len__') and len(camera_region) > 2:
            print(f"Warning: Camera {view.uid} has multiple region matches: {camera_region}. Using the first match.")
        torch.cuda.synchronize(); t0 = time.time()

        gaussians.set_anchor_mask(view.camera_center, iteration, view.resolution_scale)
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, ape_code=(int(uid_temp)), camera_region=camera_region)
        
        torch.cuda.synchronize(); t1 = time.time()
        t_list.append(t1-t0)

        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()  
        per_view_dict['{0:05d}'.format(idx)+".png"] = visible_count.item()

        # gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        # inv_depth_map = render_pkg["inv_depth"].detach().cpu().numpy()
        # # torchvision.utils.save_image(torch.tensor(inv_depth_map), os.path.join(inv_depth_path, '{0:05d}'.format(idx) + ".png"))
        # np.save(os.path.join(inv_depth_path, f"{idx}_inv_depth.npy"), inv_depth_map)
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
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, show_level : bool, ape_code : int):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
            dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
        )
        # 添加batch_size参数，支持批量加载相机
        batch_size = 300
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=dataset.resolution_scales, batch_size=batch_size)
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
            # 使用批量加载模式，每次只渲染一个批次的相机
            num_batches = scene.get_num_batches()
            for batch_idx in range(num_batches):
                batch_cameras = scene.getTrainCameras(batch_idx)
                if not batch_cameras:
                    continue
                print(f"Rendering train set, batch {batch_idx+1}/{num_batches}...")
                render_set(dataset.model_path, "train", scene.loaded_iter, batch_cameras, gaussians, pipeline, background, show_level, batch_idx)
                # 释放内存
                del batch_cameras
                torch.cuda.empty_cache()

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, show_level, ape_code)

if __name__ == "__main__":
    # Set up command line argument parser
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

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.show_level, args.ape)
    
