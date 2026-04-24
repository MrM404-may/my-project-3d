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
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None, logger=None, batch_size=None, region_camera_pools=None, lazy_load_train=False):
        """
        :param path: Path to colmap scene main folder.
        :param batch_size: Number of cameras to load per batch (None for full loading)
        :param region_camera_pools: List of camera indices for each region
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.resolution_scales = resolution_scales
        self.batch_size = batch_size
        self.args = args
        self.region_camera_pools = region_camera_pools if region_camera_pools else []

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.raw_train_cameras = []
        self.raw_test_cameras = []

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.ds)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.random_background, args.white_background,  args.eval, ply_path=ply_path)
        else:
            scene_info = sceneLoadTypeCallbacks["City"](args.source_path, args.random_background, args.white_background, args.eval, args.ds, undistorted=args.undistorted)

        self.gaussians.set_appearance(len(scene_info.train_cameras))
        
        if not self.loaded_iter:
            points = self.save_ply(scene_info.point_cloud, args.ratio, os.path.join(self.model_path, "input.ply"))
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.raw_train_cameras = scene_info.train_cameras
        self.raw_test_cameras = scene_info.test_cameras

        # 初始加载第一批相机
        if self.batch_size:
            print(f"Using batch loading with batch size: {self.batch_size}")
            self._load_camera_batch(0)
            # 加载测试相机（如果有的话）
            if scene_info.test_cameras:
                print("Loading Test Cameras")
                for resolution_scale in self.resolution_scales:
                    self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        else:
            print("Using full loading")
            for resolution_scale in self.resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        
        if self.loaded_iter:
            # 优先使用merged_anchors.ply文件
            merged_ply_path = os.path.join(self.model_path, "merged_anchors.ply")
            if os.path.exists(merged_ply_path):
                print(f"Loading merged anchors from: {merged_ply_path}")
                self.gaussians.load_ply_sparse_gaussian(merged_ply_path)
            else:
                print("Using regular point cloud ply file")
                self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                               "point_cloud",
                                                               "iteration_" + str(self.loaded_iter),
                                                               "point_cloud.ply"))
            
            # 直接使用原始保存路径下的多个PT文件
            print("Using regular MLP checkpoints")
            self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter)))
            print("Load Voxel Size: ", self.gaussians.voxel_size)
            print("Load Standard Dist: ", self.gaussians.standard_dist)
        else:
            if args.random_background:
                if logger:
                    logger.info("Using random background")
                else:
                    print("Using random background")
            elif args.white_background:
                if logger:
                    logger.info("Using white background")
                else:
                    print("Using white background")
            else:
                if logger:
                    logger.info("Using black background")
                else:
                    print("Using black background")
            points = torch.unique(points, dim=0)
            self.gaussians.set_level(points, self.train_cameras, self.resolution_scales, args.dist_ratio, args.init_level, args.levels)
            self.gaussians.create_from_pcd(points, self.cameras_extent, logger)

    def _load_camera_batch(self, batch_idx):
        """
        加载指定批次的相机
        :param batch_idx: 批次索引
        """
        if not self.batch_size:
            return
        
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.raw_train_cameras))
        
        batch_cameras = self.raw_train_cameras[start_idx:end_idx]
        
        print(f"Loading camera batch {batch_idx+1}: {start_idx} to {end_idx} of {len(self.raw_train_cameras)}")
        
        for resolution_scale in self.resolution_scales:
            if resolution_scale not in self.train_cameras:
                self.train_cameras[resolution_scale] = []
            
            # 加载当前批次的相机
            batch_camera_list = cameraList_from_camInfos(batch_cameras, resolution_scale, self.args)
            self.train_cameras[resolution_scale].extend(batch_camera_list)

    def getTrainCameras(self, batch_idx=None):
        """
        获取训练相机
        :param batch_idx: 批次索引（None表示获取所有已加载的相机）
        """
        if batch_idx is not None and self.batch_size:
            # 确保指定批次已加载
            if batch_idx * self.batch_size >= len(self.raw_train_cameras):
                return []
            
            # 检查批次是否已加载
            current_loaded = len(self.train_cameras.get(self.resolution_scales[0], []))
            if current_loaded <= batch_idx * self.batch_size:
                # 加载到指定批次
                for i in range(current_loaded // self.batch_size, batch_idx + 1):
                    self._load_camera_batch(i)
            
            # 返回指定批次的相机
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.raw_train_cameras))
            all_cams = []
            for scale in self.resolution_scales:
                all_cams.extend(self.train_cameras[scale][start_idx:end_idx])
            return all_cams
        else:
            # 返回所有已加载的相机
            all_cams = []   
            for scale in self.resolution_scales:
                all_cams.extend(self.train_cameras.get(scale, []))
            return all_cams

    def getTestCameras(self):
        all_cams = []   
        for scale in self.resolution_scales:
            all_cams.extend(self.test_cameras.get(scale, []))
        return all_cams

    def get_num_batches(self):
        """
        获取训练相机的批次数
        """
        if not self.batch_size:
            return 1
        return (len(self.raw_train_cameras) + self.batch_size - 1) // self.batch_size
    
    def load_cameras_by_ids(self, camera_ids):
        """
        根据相机ID加载特定的相机
        :param camera_ids: 相机ID列表
        """
        if not camera_ids:
            return []
        
        # 构建相机ID到原始相机信息索引的映射
        
        
        # 通过索引获取相机信息
        selected_cam_infos = [self.raw_train_cameras[idx] for idx in camera_ids]
        # test_camera_count = max(1, int(len(camera_indices) / 5))
        # selected_cam_test_infos = [self.raw_train_cameras[camera_indices[i]] for i in range(test_camera_count)]
        # selected_cam_infos = self.raw_train_cameras[camera_ids]
        # print(f"Loading {len(selected_cam_infos)} cameras by IDs: {camera_ids[:5]}{'...' if len(camera_ids) > 5 else ''}")
        
        # 加载筛选出的相机
        loaded_cameras = []
        for resolution_scale in self.resolution_scales:
            if resolution_scale not in self.train_cameras:
                self.train_cameras[resolution_scale] = []
            # if resolution_scale not in self.test_cameras:
            #     self.test_cameras[resolution_scale] = []
            
            # 加载当前批次的相机
            batch_camera_list = cameraList_from_camInfos(selected_cam_infos, resolution_scale, self.args)
            self.train_cameras[resolution_scale].extend(batch_camera_list)
            loaded_cameras.extend(batch_camera_list)
            
            # 加载测试相机
            # if selected_cam_test_infos:
            #     test_camera_list = cameraList_from_camInfos(selected_cam_test_infos, resolution_scale, self.args)
            #     self.test_cameras[resolution_scale].extend(test_camera_list)
        
        return loaded_cameras
    # def load_cameras_by_indices(self, indices):
    #     """
    #     根据序号加载特定的相机
    #     :param indices: 相机序号列表
    #     """
    #     if not indices:
    #         return []
        
    #     # 确保所有指定序号的相机都已加载
    #     max_idx = max(indices)
    #     if max_idx >= len(self.raw_train_cameras):
    #         print(f"警告: 序号 {max_idx} 超出相机范围")
    #         indices = [idx for idx in indices if idx < len(self.raw_train_cameras)]
        
    #     # 计算需要加载的批次
    #     if self.batch_size:
    #         max_batch = max_idx // self.batch_size
    #         current_loaded = len(self.train_cameras.get(self.resolution_scales[0], []))
    #         current_batch = current_loaded // self.batch_size
            
    #         # 加载需要的批次
    #         for i in range(current_batch, max_batch + 1):
    #             self._load_camera_batch(i)
        
    #     # 收集指定序号的相机
    #     loaded_cameras = []
    #     for scale in self.resolution_scales:
    #         scale_cameras = self.train_cameras.get(scale, [])
    #         for idx in indices:
    #             if idx < len(scale_cameras):
    #                 cam = scale_cameras[idx]
    #                 # 为相机设置region_idx属性
    #                 # 使用camera_id_to_region映射来获取相机所属的区域
    #                 cam.region_idx = 0  # 默认区域0
    #                 if hasattr(self, 'camera_id_to_region'):
    #                     # 尝试将idx转换为字符串，因为camera_id_to_region的键可能是字符串
    #                     cam_id_str = str(idx)
    #                     if cam_id_str in self.camera_id_to_region:
    #                         cam.region_idx = self.camera_id_to_region[cam_id_str]
    #                 loaded_cameras.append(cam)
        
    #     return loaded_cameras

    def release_images(self, indices=None):
        """
        释放图像内存
        :param indices: 相机序号列表，None表示释放所有训练相机的图像
        """
        if indices is None:
            # 只释放训练相机的图像，保留测试相机的图像
            for scale in self.resolution_scales:
                scale_cameras = self.train_cameras.get(scale, [])
                for cam in scale_cameras:
                    # 释放Camera对象的original_image
                    if hasattr(cam, 'original_image'):
                        cam.original_image = None
                    # 释放CameraInfo对象的_image
                    elif hasattr(cam, 'camera_info') and hasattr(cam.camera_info, 'release_image'):
                        cam.camera_info.release_image()
                    elif hasattr(cam, 'release_image'):
                        cam.release_image()
        else:
            # 释放指定序号的相机图像
            for scale in self.resolution_scales:
                scale_cameras = self.train_cameras.get(scale, [])
                for idx in indices:
                    if idx < len(scale_cameras):
                        cam = scale_cameras[idx]
                        # 释放Camera对象的original_image
                        if hasattr(cam, 'original_image'):
                            cam.original_image = None
                        # 释放CameraInfo对象的_image
                        elif hasattr(cam, 'camera_info') and hasattr(cam.camera_info, 'release_image'):
                            cam.camera_info.release_image()
                        elif hasattr(cam, 'release_image'):
                            cam.release_image()

    def save_ply(self, pcd, ratio, path):
        points = torch.tensor(pcd.points[::ratio]).float().cuda()
        colors = torch.tensor(pcd.colors[::ratio]).float().cuda()
        storePly(path, points.cpu().numpy(), colors.cpu().numpy())
        return points

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)