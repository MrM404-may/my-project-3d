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
from scene.cameras import Camera, MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Renderer:
    def __init__(self):
        self.gaussians = None
        self.cameras_json = None
        self.cache_json = None
        self.regions_config = None
        self.camera_id_to_region = {}
        self.background = None
        self.pipeline = None
        self.dataset = None
        self.iteration = None

    def render_init(self, dataset, iteration, pipeline, source_path):
        """
        初始化渲染器：加载PLY模型、cameras.json、cache.json和regions_config.json
        
        参数:
            dataset: 数据集参数
            iteration: 迭代次数
            pipeline: 渲染管道参数
            source_path: 源数据路径
        """
        # 完全复制自render9-huang.py的初始化过程
        with torch.no_grad():
            # 加载Gaussian模型
            print(f"Loading trained model at iteration {iteration}")
            self.gaussians = GaussianModel(
                dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim,
                dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level,
                dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
            )
            
            # 创建Scene对象
            scene = Scene(dataset, self.gaussians, load_iteration=iteration, shuffle=False, resolution_scales=dataset.resolution_scales)
            self.gaussians.eval()
            self.gaussians.plot_levels()
            self.iteration = scene.loaded_iter
            
            # 设置背景
            if dataset.random_background:
                bg_color = [np.random.random(), np.random.random(), np.random.random()]
            elif dataset.white_background:
                bg_color = [1.0, 1.0, 1.0]
            else:
                bg_color = [0.0, 0.0, 0.0]
            self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
            # 确保模型路径存在
            if not os.path.exists(dataset.model_path):
                os.makedirs(dataset.model_path)
            
            # 保存参数
            self.dataset = dataset
            self.pipeline = pipeline

        # 加载区域配置和相机ID到区域的映射
        regions_config_path = os.path.join(source_path, "regions_config.json")
        if os.path.exists(regions_config_path):
            with open(regions_config_path, 'r', encoding='utf-8') as f:
                self.regions_config = json.load(f)
            
            # 加载相机ID到区域的映射
            cache_json_path = os.path.join(source_path, "cache.json")
            if os.path.exists(cache_json_path):
                with open(cache_json_path, 'r', encoding='utf-8') as f:
                    camera_info_dict = json.load(f)

                # 核心匹配逻辑
                print("===== 开始匹配相机与区域 =====")
                for cam_id, cam_data in camera_info_dict.items():
                    if cam_id not in self.camera_id_to_region:
                        for region_idx, region in enumerate(self.regions_config):
                            region_name = region['name']
                            if cam_data['in_regions'].get(region_name, False) and (int(cam_id) < int(region['max_id'])):
                                self.camera_id_to_region[cam_id] = region_idx
                                break
                 
                # 统计各区域详情
                print("\n===== 各区域详细信息统计表 =====")
                region_stats = {i: [0, -1] for i in range(len(self.regions_config))}
                for cam_id, region_idx in self.camera_id_to_region.items():
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
                print(f"错误：缓存文件不存在！路径：{cache_json_path}")
        else:
            print(f"错误：区域配置文件不存在！路径：{regions_config_path}")

        # 加载cameras.json
        cameras_json_path = os.path.join(source_path, "cameras.json")
        if os.path.exists(cameras_json_path):
            with open(cameras_json_path, 'r', encoding='utf-8') as f:
                self.cameras_json = json.load(f)
            # 确保cameras_json是字典格式
            if isinstance(self.cameras_json, list):
                # 将列表转换为字典，以id为键
                cameras_dict = {}
                for cam in self.cameras_json:
                    cam_id = str(cam.get('id', cam.get('uid', len(cameras_dict))))
                    cameras_dict[cam_id] = cam
                self.cameras_json = cameras_dict
        else:
            raise FileNotFoundError(f"cameras.json not found at {cameras_json_path}")

        print("✅ Renderer initialized successfully!")

    def quaternion_to_rotation_matrix(self, q):
        """
        将四元数转换为旋转矩阵
        
        参数:
            q: 四元数 [w, x, y, z]
        
        返回:
            3x3旋转矩阵
        """
        w, x, y, z = q
        
        # 计算旋转矩阵
        rot_matrix = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        
        return rot_matrix

    def find_most_similar_camera(self, position, quaternion):
        """
        找到与输入相机位姿最相似的训练相机
        
        参数:
            position: 相机位置 [x, y, z]
            quaternion: 相机姿态四元数 [w, x, y, z]
        
        返回:
            最相似的相机ID, 区域, ape_code
        """
        if not self.cameras_json:
            raise ValueError("cameras.json not loaded")
        
        # 转换四元数为旋转矩阵
        rot_matrix = self.quaternion_to_rotation_matrix(quaternion)
        
        # 计算与每个训练相机的相似度
        best_score = float('inf')
        best_camera_id = None
        
        for cam_id, cam_data in self.cameras_json.items():
            # 获取相机位置和旋转
            cam_position = np.array(cam_data.get('position', [0, 0, 0]))
            cam_rotation = np.array(cam_data.get('rotation', [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            
            # 计算位置距离
            position_distance = np.linalg.norm(np.array(position) - cam_position)
            
            # 计算旋转相似度（使用矩阵余弦相似度）
            rotation_similarity = np.abs(np.trace(np.dot(rot_matrix, cam_rotation.T)) / 3)
            rotation_distance = 1 - rotation_similarity
            
            # 综合评分（位置距离 + 旋转距离）
            score = position_distance + rotation_distance
            
            if score < best_score:
                best_score = score
                best_camera_id = cam_id
        
        # 获取区域和ape_code
        region = self.camera_id_to_region.get(best_camera_id, 0)
        ape_code = 0  # 暂时使用0，实际应该从cache.json中获取
        
        return best_camera_id, region, ape_code

    def render(self, position, quaternion, fov=90.0):
        """
        渲染新视角
        
        参数:
            position: 相机位置 [x, y, z]
            quaternion: 相机姿态四元数 [w, x, y, z]
            fov: 视场角，默认90度
        
        返回:
            渲染结果图像
        """
        if not self.gaussians:
            raise ValueError("Renderer not initialized")
        
        # 找到最相似的训练相机
        camera_id, region, ape_code = self.find_most_similar_camera(position, quaternion)
        print(f"📸 Most similar camera: ID={camera_id}, region={region}, ape_code={ape_code}")
        
        # 转换四元数为旋转矩阵
        rot_matrix = self.quaternion_to_rotation_matrix(quaternion)
        
        # 计算视图矩阵
        world_view_transform = getWorld2View2(torch.tensor(rot_matrix), torch.tensor(position), transposed=True)
        world_view_transform = world_view_transform.cuda()
        
        # 计算投影矩阵
        fx = fy = 1.0 / (2 * np.tan(fov * np.pi / 360))
        projection_matrix = getProjectionMatrix(znear=0.01, zfar=100, fovX=fov, fovY=fov).cuda()
        
        # 渲染
        with torch.no_grad():
            rendering = render(
                self.gaussians,  # 这里传入self.gaussians，不需要传入region
                world_view_transform, 
                projection_matrix, 
                self.background, 
                self.pipeline,
                region=region  # 这里传入region
            )
        
        return rendering['render']


def render_sets(dataset, iteration, pipeline, skip_train, skip_test, show_level, ape):
    """
    渲染测试集
    """
    # 初始化渲染器
    renderer = Renderer()
    
    # 初始化
    renderer.render_init(dataset, iteration, pipeline, dataset.source_path)
    
    # 测试渲染
    print("🔍 Testing with camera 0")
    if renderer.cameras_json and "0" in renderer.cameras_json:
        test_cam = renderer.cameras_json["0"]
        test_position = test_cam.get("position", [0, 0, 0])
        test_rotation = test_cam.get("rotation", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        # 从旋转矩阵计算四元数
        def rotation_matrix_to_quaternion(R):
            trace = np.trace(R)
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                w = 0.25 / s
                x = (R[2, 1] - R[1, 2]) * s
                y = (R[0, 2] - R[2, 0]) * s
                z = (R[1, 0] - R[0, 1]) * s
            else:
                if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                    w = (R[2, 1] - R[1, 2]) / s
                    x = 0.25 * s
                    y = (R[0, 1] + R[1, 0]) / s
                    z = (R[0, 2] + R[2, 0]) / s
                elif R[1, 1] > R[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                    w = (R[0, 2] - R[2, 0]) / s
                    x = (R[0, 1] + R[1, 0]) / s
                    y = 0.25 * s
                    z = (R[1, 2] + R[2, 1]) / s
                else:
                    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                    w = (R[1, 0] - R[0, 1]) / s
                    x = (R[0, 2] + R[2, 0]) / s
                    y = (R[1, 2] + R[2, 1]) / s
                    z = 0.25 * s
            return [w, x, y, z]
        
        test_quaternion = rotation_matrix_to_quaternion(test_rotation)
        print(f"   Position: {test_position}")
        print(f"   Quaternion: {test_quaternion}")
        
        # 渲染
        rendered_image = renderer.render(test_position, test_quaternion)
        print(rendered_image)
        
        # 保存结果
        torchvision.utils.save_image(rendered_image, "test_render.png")
        print("✅ Rendered image saved to test_render.png")
    else:
        print("❌ Camera 0 not found in cameras.json")


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