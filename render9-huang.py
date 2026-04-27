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
import json
import time
from tqdm import tqdm

# 导入必要的模块
from scene import Scene
from gaussian_renderer import render, prefilter_voxel
import torchvision
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


class Renderer:
    def __init__(self, model_path, data_path, iteration=-1, regions_config_path=None, cache_path=None, cameras_json_path=None):
        """
        初始化 Renderer
        model_path: 模型输出路径
        data_path: 数据路径
        iteration: 迭代次数
        regions_config_path: 区域配置文件路径
        cache_path: 缓存文件路径
        cameras_json_path: 相机数据文件路径
        """
        self.model_path = model_path
        self.data_path = data_path
        self.iteration = iteration
        self.regions_config_path = regions_config_path or "/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma0422/regions_config.json"
        self.cache_path = cache_path or "/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma0422/cache.json"
        self.cameras_json_path = cameras_json_path or os.path.join(data_path, "cameras.json")
        self.camera_id_to_region = {}
        self.regions_config = []
        self.cameras_data = []
        
        # 初始化模型和场景
        from arguments import ModelParams, PipelineParams, get_combined_args
        import argparse
        import sys
        
        # 保存原始命令行参数
        original_argv = sys.argv.copy()
        
        # 创建参数解析器
        parser = argparse.ArgumentParser(description="Testing script parameters")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        parser.add_argument("--iteration", default=-1, type=int)
        
        # 临时设置命令行参数
        sys.argv = [
            "render9-huang.py",
            "--model_path", model_path,
            "--source_path", data_path,
            "--iteration", str(iteration)
        ]
        
        # 获取完整的参数（包括从配置文件读取的）
        args = get_combined_args(parser)
        
        # 恢复原始命令行参数
        sys.argv = original_argv
        
        # 提取参数
        self.dataset = model.extract(args)
        self.pipeline = pipeline.extract(args)
        
        # 加载 cameras.json 文件
        self.load_cameras_data()
        
        # 初始化 Gaussian 和 Scene
        self.gaussians = GaussianModel(
            self.dataset.feat_dim, self.dataset.n_offsets, self.dataset.fork, self.dataset.use_feat_bank, self.dataset.appearance_dim, 
            self.dataset.add_opacity_dist, self.dataset.add_cov_dist, self.dataset.add_color_dist, self.dataset.add_level, 
            self.dataset.visible_threshold, self.dataset.dist2level, self.dataset.base_layer, self.dataset.progressive, self.dataset.extend
        )
        
        self.scene = Scene(self.dataset, self.gaussians, load_iteration=iteration, shuffle=False, resolution_scales=self.dataset.resolution_scales)
        self.gaussians.eval()
        self.gaussians.plot_levels()
        
        # 打印锚点的region属性信息
        print(f"锚点总数: {self.gaussians._anchor.shape[0]}")
        print(f"锚点region唯一值: {torch.unique(self.gaussians._region)}")
        for region_val in torch.unique(self.gaussians._region):
            count = (self.gaussians._region == region_val).sum().item()
            print(f"  region {region_val.item()}: {count} 个锚点")
        
        # 初始化背景
        if self.dataset.random_background:
            bg_color = [np.random.random(),np.random.random(),np.random.random()] 
        elif self.dataset.white_background:
            bg_color = [1.0, 1.0, 1.0]
        else:
            bg_color = [0.0, 0.0, 0.0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 创建输出目录
        if not os.path.exists(self.dataset.model_path):
            os.makedirs(self.dataset.model_path)
    
    def load_cameras_data(self):
        """
        加载 cameras.json 文件
        """
        if os.path.exists(self.cameras_json_path):
            try:
                with open(self.cameras_json_path, 'r', encoding='utf-8') as f:
                    self.cameras_data = json.load(f)
                print(f"✅ 成功加载 cameras.json 文件，共 {len(self.cameras_data)} 个相机")
            except Exception as e:
                print(f"❌ 加载 cameras.json 文件失败: {e}")
                self.cameras_data = []
        else:
            print(f"❌ cameras.json 文件不存在: {self.cameras_json_path}")
            self.cameras_data = []
    
    def find_similar_camera(self, R, T):
        """
        找到与输入 R 和 T 最相似的相机
        R: 旋转矩阵
        T: 平移向量
        返回: (最相似的相机ID, 对应的region, ape_code)
        """
        if not self.cameras_data:
            print("⚠️  没有加载相机数据")
            return None, None, None
        
        if not self.camera_id_to_region:
            self.load_region_config()
        
        # 转换输入的 R 和 T 为 numpy 数组
        R_input = np.array(R)
        T_input = np.array(T)
        
        min_distance = float('inf')
        best_camera = None
        
        for camera in self.cameras_data:
            try:
                # 获取相机的位置和旋转
                cam_position = np.array(camera['position'])
                cam_rotation = np.array(camera['rotation'])
                
                # 计算位置距离
                position_distance = np.linalg.norm(T_input - cam_position)
                
                # 计算旋转相似度（使用矩阵余弦相似度）
                rotation_similarity = np.sum(R_input * cam_rotation) / (np.linalg.norm(R_input) * np.linalg.norm(cam_rotation))
                rotation_distance = 1 - rotation_similarity
                
                # 综合距离（位置距离 + 旋转距离）
                total_distance = position_distance + rotation_distance
                
                if total_distance < min_distance:
                    min_distance = total_distance
                    best_camera = camera
            except Exception as e:
                continue
        
        if best_camera:
            camera_id = str(best_camera['id'])
            region = self.camera_id_to_region.get(camera_id, 0)
            ape_code = best_camera['id']  # id 对应 ape_code
            print(f"✅ 找到最相似的相机: ID={camera_id}, 区域={region}, APE代码={ape_code}")
            return camera_id, region, ape_code
        else:
            print("⚠️  没有找到相似的相机")
            return None, 0, 10  # 返回默认值
    
    def load_region_config(self):
        """
        加载区域配置和相机ID到区域的映射
        """
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
        """
        渲染一组视图
        """
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        os.makedirs(render_path, exist_ok=True)
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

        inv_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "inv_depth")
        os.makedirs(inv_depth_path, exist_ok=True)

        os.makedirs(gts_path, exist_ok=True)
        if show_level:
            render_level_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_level")
            os.makedirs(render_level_path, exist_ok=True)

        if not self.camera_id_to_region:
            self.load_region_config()

        t_list = []
        per_view_dict = {}
        per_view_level_dict = {}
        all_cameras_info = []
        camera_region = 0
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            # 尝试从视图的属性中获取区域信息
            if hasattr(view, 'camera_region'):
                camera_region = view.camera_region
            else:
                # 否则从相机ID映射中获取
                camera_region = self.camera_id_to_region.get(str(getattr(view, 'uid', '')), 0)
            
            # 检查当前区域是否有锚点
            region_anchors = (gaussians._region == camera_region).sum().item()
            print(f"正在渲染视图 {idx}, 相机ID: {getattr(view, 'uid', 'N/A')}, 区域: {camera_region}, APE代码: {ape_code}, 锚点数量: {region_anchors}")
            
            # 如果当前区域没有锚点，使用所有锚点（设置为区域 -1）
            if region_anchors == 0:
                print("⚠️  当前区域没有锚点，使用所有锚点")
                camera_region = -1
            
            if hasattr(camera_region, '__len__') and len(camera_region) > 2:
                print(f"Warning: Camera {view.uid} has multiple region matches: {camera_region}. Using the first match.")
            torch.cuda.synchronize(); t0 = time.time()

            gaussians.set_anchor_mask(view.camera_center, iteration, view.resolution_scale)
            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
            render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, ape_code=ape_code, camera_region=camera_region)
            
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
    
    def render_set_one_view(self, R, T, show_level : bool = False, ape_code : int = None):
        """
        渲染单个视图，只需要输入旋转矩阵 R 和平移向量 T
        R: 旋转矩阵
        T: 平移向量
        show_level: 是否显示不同层级
        ape_code: 外观编码（如果为 None，则自动从相似相机中获取）
        """
        with torch.no_grad():
            # 找到最相似的相机
            _, camera_region, auto_ape_code = self.find_similar_camera(R, T)
            
            # 使用找到的 ape_code 或默认值
            final_ape_code = ape_code if ape_code is not None else auto_ape_code
            
            # 检查当前区域是否有锚点
            region_anchors = (self.gaussians._region == camera_region).sum().item()
            print(f"区域 {camera_region} 的锚点数量: {region_anchors}")
            
            # 如果当前区域没有锚点，使用所有锚点（设置为区域 -1）
            if region_anchors == 0:
                print("⚠️  当前区域没有锚点，使用所有锚点")
                camera_region = -1
            
            # 尝试获取测试相机作为模板
            view = None
            test_cameras = self.scene.getTestCameras()
            if test_cameras:
                view = test_cameras[0]
                print("✅ 使用测试相机作为模板")
            else:
                # 如果没有测试相机，尝试使用训练相机
                train_cameras = self.scene.getTrainCameras()
                if train_cameras:
                    view = train_cameras[0]
                    print("✅ 使用训练相机作为模板")
                else:
                    print("Error: No cameras available")
                    return
            
            # 设置相机的旋转矩阵和平移向量
            view.R = R
            view.T = T
            # 设置相机的区域信息
            view.camera_region = camera_region
            
            # 重新计算相机的位姿相关属性
            import numpy as np
            from utils.graphics_utils import getWorld2View2, getProjectionMatrix
            # 将 R 和 T 转换为 numpy 数组
            R_np = np.array(R)
            T_np = np.array(T)
            
            # 使用默认的 translate 和 scale 参数，而不是 view.trans 和 view.scale
            # 这样可以确保计算结果只依赖于输入的 R 和 T
            view.world_view_transform = torch.tensor(getWorld2View2(R_np, T_np)).transpose(0, 1).cuda()
            view.projection_matrix = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy).transpose(0,1).cuda()
            view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
            view.camera_center = view.world_view_transform.inverse()[3, :3]
            
            # 渲染单个视图
            self.render_set(
                self.dataset.model_path, 
                "test", 
                self.scene.loaded_iter, 
                [view], 
                self.gaussians, 
                self.pipeline, 
                self.background, 
                show_level, 
                final_ape_code
            )
    
    def render_sets(self, skip_train : bool = False, skip_test : bool = False, show_level : bool = False, ape_code : int = 10):
        """
        渲染训练集和测试集
        skip_train: 是否跳过训练集
        skip_test: 是否跳过测试集
        show_level: 是否显示不同层级
        ape_code: 外观编码
        """
        with torch.no_grad():
            ape_code = 10
            if not skip_train:
                num_batches = 588
                for batch_idx in range(num_batches):
                    batch_cameras = self.scene.get_unloaded_cameras(range(num_batches))
                    if not batch_cameras:
                        continue
                    print(f"Rendering train set, batch {batch_idx+1}/{num_batches}...")
                    self.render_set(
                        self.dataset.model_path, 
                        "train", 
                        self.scene.loaded_iter, 
                        batch_cameras, 
                        self.gaussians, 
                        self.pipeline, 
                        self.background, 
                        show_level, 
                        batch_idx
                    )
                    del batch_cameras
                    torch.cuda.empty_cache()

            if not skip_test:
                self.render_set(
                    self.dataset.model_path, 
                    "test", 
                    self.scene.loaded_iter, 
                    self.scene.getTestCameras(), 
                    self.gaussians, 
                    self.pipeline, 
                    self.background, 
                    show_level, 
                    ape_code
                )
