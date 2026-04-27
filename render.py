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
from scene import Scene
from gaussian_renderer import render, prefilter_voxel
import torchvision
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel
from utils.camera_utils import Camera

class Renderer:
    def __init__(self, model_path, source_path):
        self.model_path = model_path
        self.source_path = source_path
        self.scene = None
        self.gaussians = None
        self.pipeline = None
        self.background = None
        self.cameras_data = []
        self.camera_id_to_region = {}
        self.camera_positions = []
        self.camera_rotations = []
        self.camera_uids = []
        self.camera_dict = {}
    
    def render_init(self):
        """
        Load PLY model, cameras.json, and cache.json files
        Load one image
        """
        # Initialize system state (RNG)
        safe_state(True)
        
        # Create model parameters directly
        class ModelParams:
            def __init__(self):
                self.feat_dim = 32
                self.n_offsets = 16
                self.fork = 1
                self.use_feat_bank = False
                self.images = "images"
                self.resolution = -1
                self.white_background = False
                self.random_background = False
                self.resolution_scales = [1.0]
                self.data_device = "cuda"
                self.eval = False
                self.ds = 1
                self.ratio = 1
                self.undistorted = False
                self.appearance_dim = 0
                self.add_opacity_dist = False
                self.add_cov_dist = False
                self.add_color_dist = False
                self.add_level = True
                self.extend = 1.1
                self.dist2level = "progressive"
                self.base_layer = -1
                self.visible_threshold = 0.0
                self.update_ratio = 0.2
                self.progressive = False
                self.dist_ratio = 0.999
                self.levels = -1
                self.init_level = -1
                self.extra_ratio = 0.25
                self.extra_up = 0.01
        
        # Create pipeline parameters directly
        class PipelineParams:
            def __init__(self):
                self.compute_cov3D_python = False
                self.debug = False
        
        # Create parameters objects
        model_params = ModelParams()
        model_params.source_path = self.source_path
        model_params.model_path = self.model_path
        
        pipeline_params = PipelineParams()
        self.pipeline = pipeline_params
        
        # Initialize Gaussian model
        self.gaussians = GaussianModel(
            model_params.feat_dim, model_params.n_offsets, model_params.fork, model_params.use_feat_bank, model_params.appearance_dim, 
            model_params.add_opacity_dist, model_params.add_cov_dist, model_params.add_color_dist, model_params.add_level, 
            model_params.visible_threshold, model_params.dist2level, model_params.base_layer, model_params.progressive, model_params.extend
        )
        
        # Load scene
        self.scene = Scene(model_params, self.gaussians, load_iteration=-1, shuffle=False, resolution_scales=[1.0])
        self.gaussians.eval()
        
        # Set background color
        if model_params.random_background:
            bg_color = [np.random.random(), np.random.random(), np.random.random()]
        elif model_params.white_background:
            bg_color = [1.0, 1.0, 1.0]
        else:
            bg_color = [0.0, 0.0, 0.0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Load cameras.json
        cameras_json_path = os.path.join(self.source_path, "cameras.json")
        if os.path.exists(cameras_json_path):
            with open(cameras_json_path, 'r') as f:
                self.cameras_data = json.load(f)
        
        # Load cache.json and build camera_id_to_region mapping
        cache_path = os.path.join(self.source_path, "cache.json")
        regions_config_path = os.path.join(self.source_path, "regions_config.json")
        
        if os.path.exists(cache_path) and os.path.exists(regions_config_path):
            with open(regions_config_path, 'r', encoding='utf-8') as f:
                regions_config = json.load(f)
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                camera_info_dict = json.load(f)
            
            # Build camera_id_to_region mapping
            for cam_id, cam_data in camera_info_dict.items():
                if cam_id not in self.camera_id_to_region:
                    for region_idx, region in enumerate(regions_config):
                        region_name = region['name']
                        if cam_data['in_regions'].get(region_name, False) and (int(cam_id) < int(region['max_id'])):
                            self.camera_id_to_region[cam_id] = region_idx
                            break
        
        # Extract camera positions and rotations for similarity calculation
        for cam_data in self.cameras_data:
            self.camera_positions.append(np.array(cam_data['position']))
            # Convert 3x3 rotation matrix to quaternion for easier similarity calculation
            self.camera_rotations.append(self.rotation_matrix_to_quaternion(np.array(cam_data['rotation'])))
            self.camera_uids.append(cam_data['id'])
        
        # Store all camera data for easy access
        self.camera_dict = {cam_data['id']: cam_data for cam_data in self.cameras_data}
        
        # Load one image (first camera)
        if self.scene.train_cameras:
            first_camera = next(iter(self.scene.train_cameras.values()))[0]
            if hasattr(first_camera, 'load_image_to_gpu'):
                first_camera.load_image_to_gpu()
        
        print("Render initialization completed successfully")
    
    def find_most_similar_camera(self, position, rotation):
        """
        Find the most similar camera from the training set based on position and rotation
        """
        min_distance = float('inf')
        most_similar_idx = 0
        
        for i, (cam_pos, cam_rot) in enumerate(zip(self.camera_positions, self.camera_rotations)):
            # Calculate position distance
            pos_dist = np.linalg.norm(np.array(position) - cam_pos)
            
            # Calculate rotation distance (using quaternion dot product)
            rot_dot = np.abs(np.dot(np.array(rotation), cam_rot))
            rot_dist = 1.0 - rot_dot  # Closer to 0 means more similar
            
            # Combined distance
            combined_dist = pos_dist + rot_dist * 10.0  # Weight rotation similarity
            
            if combined_dist < min_distance:
                min_distance = combined_dist
                most_similar_idx = i
        
        return self.camera_uids[most_similar_idx]
    
    def quaternion_to_rotation_matrix(self, quaternion):
        """
        Convert quaternion [w, x, y, z] to 3x3 rotation matrix
        """
        w, x, y, z = quaternion
        
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        return R
    
    def rotation_matrix_to_quaternion(self, R):
        """
        Convert 3x3 rotation matrix to quaternion [w, x, y, z]
        """
        # Ensure the rotation matrix is a numpy array
        R = np.array(R)
        
        # Calculate trace of the matrix
        trace = np.trace(R)
        
        if trace > 0:
            s = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
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
        
        return np.array([w, x, y, z])
    
    def render_new(self, position, rotation, output_path):
        """
        Render a new viewpoint given camera position (xyz) and orientation (quaternion)
        """
        # Find most similar camera
        similar_camera_uid = self.find_most_similar_camera(position, rotation)
        
        # Get region for the similar camera
        camera_region = self.camera_id_to_region.get(str(similar_camera_uid), 0)
        
        # Get APE code (using camera UID as APE code)
        ape_code = similar_camera_uid
        
        # Create a new camera with the given position and rotation
        # Note: We need to create a Camera object with the appropriate parameters
        # For simplicity, we'll use the first camera as a template
        template_camera = next(iter(self.scene.train_cameras.values()))[0]
        
        # Convert quaternion to rotation matrix
        rotation_matrix = self.quaternion_to_rotation_matrix(rotation)
        
        # Create new camera
        new_camera = Camera(
            colmap_id=similar_camera_uid,
            R=rotation_matrix,
            T=position,
            FoVx=template_camera.FoVx,
            FoVy=template_camera.FoVy,
            image_path=template_camera.image_path,
            gt_alpha_mask_path=getattr(template_camera, 'gt_alpha_mask_path', None),
            image_name=getattr(template_camera, 'image_name', f'new_view_{similar_camera_uid}'),
            resolution_scale=template_camera.resolution_scale,
            uid=similar_camera_uid,
            data_device="cuda"
        )
        
        # Render the new viewpoint
        torch.cuda.synchronize()
        t0 = time.time()
        
        self.gaussians.set_anchor_mask(new_camera.camera_center, self.scene.loaded_iter, 1.0)
        voxel_visible_mask = prefilter_voxel(new_camera, self.gaussians, self.pipeline, self.background)
        render_pkg = render(new_camera, self.gaussians, self.pipeline, self.background, visible_mask=voxel_visible_mask, ape_code=ape_code, camera_region=camera_region)
        
        torch.cuda.synchronize()
        t1 = time.time()
        
        # Save the rendered image
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchvision.utils.save_image(rendering, output_path)
        
        print(f"Render completed in {t1 - t0:.4f} seconds")
        print(f"Rendered image saved to: {output_path}")
        print(f"Most similar camera: {similar_camera_uid}")
        print(f"Camera region: {camera_region}")
        print(f"APE code: {ape_code}")
    
    def render_training_camera(self, camera_id, output_path):
        """
        Render a specific training camera by ID
        """
        if camera_id not in self.camera_dict:
            raise ValueError(f"Camera ID {camera_id} not found in cameras data")
        
        cam_data = self.camera_dict[camera_id]
        
        # Get region for this camera
        camera_region = self.camera_id_to_region.get(str(camera_id), 0)
        
        # Use camera ID as APE code
        ape_code = camera_id
        
        # Get the actual camera object from the scene if available
        actual_camera = None
        if hasattr(self.scene, 'camera_id_map') and camera_id in self.scene.camera_id_map:
            actual_camera = self.scene.camera_id_map[camera_id]
        else:
            # Fallback to first camera as template
            actual_camera = next(iter(self.scene.train_cameras.values()))[0]
        
        # Create new camera from cameras.json data
        position = np.array(cam_data['position'])
        rotation_matrix = np.array(cam_data['rotation'])
        
        # Note: The Camera class expects R as a 3x3 matrix and T as a translation vector
        # We need to make sure we're passing them in the correct format
        new_camera = Camera(
            colmap_id=camera_id,
            R=rotation_matrix,
            T=position,
            FoVx=actual_camera.FoVx,
            FoVy=actual_camera.FoVy,
            image_path=actual_camera.image_path,
            gt_alpha_mask_path=getattr(actual_camera, 'gt_alpha_mask_path', None),
            image_name=cam_data['img_name'],
            resolution_scale=actual_camera.resolution_scale,
            uid=camera_id,
            data_device="cuda"
        )
        
        # Render the viewpoint
        torch.cuda.synchronize()
        t0 = time.time()
        
        self.gaussians.set_anchor_mask(new_camera.camera_center, self.scene.loaded_iter, 1.0)
        voxel_visible_mask = prefilter_voxel(new_camera, self.gaussians, self.pipeline, self.background)
        render_pkg = render(new_camera, self.gaussians, self.pipeline, self.background, visible_mask=voxel_visible_mask, ape_code=ape_code, camera_region=camera_region)
        
        torch.cuda.synchronize()
        t1 = time.time()
        
        # Save the rendered image
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchvision.utils.save_image(rendering, output_path)
        
        print(f"Render completed in {t1 - t0:.4f} seconds")
        print(f"Rendered image saved to: {output_path}")
        print(f"Camera ID: {camera_id}")
        print(f"Image name: {cam_data['img_name']}")
        print(f"Camera region: {camera_region}")
        print(f"APE code: {ape_code}")

if __name__ == "__main__":
    # Example usage
    renderer = Renderer(
        model_path="/root/autodl-tmp/Octree-GS/Octree-GS/output/Ma0422",
        source_path="/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma0422"
    )
    
    # Initialize renderer
    renderer.render_init()
    
    # Example: Render a new viewpoint
    # position = [x, y, z] in global coordinate system
    # rotation = [w, x, y, z] quaternion
    test_position = [0.0, 0.0, 0.0]
    test_rotation = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
    output_file = "./output_render.png"
    
    renderer.render_new(test_position, test_rotation, output_file)
