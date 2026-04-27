
import os
import torch
import numpy as np
import json
from scene import Scene
from gaussian_renderer import render, prefilter_voxel
from gaussian_renderer import GaussianModel
from scene.cameras import Camera, MiniCam
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams
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

    def render_init(self, model_path, cameras_json_path, cache_json_path, regions_config_path, iteration=-1, white_background=False):
        """
        初始化渲染器：加载PLY模型、cameras.json、cache.json和regions_config.json
        
        参数:
            model_path: 模型路径（含PLY文件）
            cameras_json_path: cameras.json文件路径
            cache_json_path: cache.json文件路径
            regions_config_path: regions_config.json文件路径
            iteration: 迭代次数，默认-1
            white_background: 是否使用白色背景
        """
        # 1. 加载dataset配置（使用原始代码结构）
        from argparse import ArgumentParser
        parser = ArgumentParser()
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        
        # 2. 创建args对象
        class DummyArgs:
            def __init__(self):
                self.model_path = model_path
                self.source_path = model_path
                self.images = "images"
                self.resolution = -1
                self.white_background = white_background
                self.random_background = False
                self.data_device = "cuda"
                self.eval = False
                self.sh_degree = 3
                self.feat_dim = 32
                self.n_offsets = 10
                self.fork = 4
                self.use_feat_bank = False
                self.appearance_dim = 32
                self.add_opacity_dist = False
                self.add_cov_dist = False
                self.add_color_dist = False
                self.add_level = False
                self.visible_threshold = 3
                self.dist2level = 'round'
                self.base_layer = -1
                self.progressive = False
                self.extend = 1.1
                self.resolution_scales = [1.0]
                self.undistorted = False
                self.ratio = 1.0
                self.ds = 1

        args = DummyArgs()
        
        # 3. 提取参数
        self.dataset = model.extract(args)
        self.pipeline = pipeline.extract(args)
        
        # 4. 设置pipeline参数
        self.pipeline.convert_SHs_python = False
        self.pipeline.compute_cov3D_python = False
        self.pipeline.debug = False

        # 5. 直接加载Gaussian模型（绕过Scene的相机加载）
        from gaussian_renderer import GaussianModel
        
        print(f"Loading trained model at iteration {iteration}")
        self.gaussians = GaussianModel(
            self.dataset.feat_dim, self.dataset.n_offsets, self.dataset.fork, self.dataset.use_feat_bank,
            self.dataset.appearance_dim, self.dataset.add_opacity_dist, self.dataset.add_cov_dist,
            self.dataset.add_color_dist, self.dataset.add_level, self.dataset.visible_threshold,
            self.dataset.dist2level, self.dataset.base_layer, self.dataset.progressive, self.dataset.extend
        )
        
        # 直接加载模型文件，绕过Scene的相机加载
        if iteration == -1:
            # 查找最新的迭代
            import glob
            import os
            iteration_dirs = glob.glob(os.path.join(model_path, "point_cloud", "iteration_*"))
            if not iteration_dirs:
                raise FileNotFoundError(f"No iteration directories found in {os.path.join(model_path, 'point_cloud')}")
            iterations = [int(d.split('_')[-1]) for d in iteration_dirs]
            self.iteration = max(iterations)
        else:
            self.iteration = iteration
        
        # 加载点云
        ply_path = os.path.join(model_path, "point_cloud", f"iteration_{self.iteration}", "point_cloud.ply")
        if not os.path.exists(ply_path):
            # 尝试加载merged_anchors.ply
            ply_path = os.path.join(model_path, "merged_anchors.ply")
            if not os.path.exists(ply_path):
                raise FileNotFoundError(f"No PLY file found at {ply_path}")
        
        self.gaussians.load_ply_sparse_gaussian(ply_path)
        
        # 加载MLP checkpoint
        mlp_path = os.path.join(model_path, "point_cloud", f"iteration_{self.iteration}")
        if not os.path.exists(mlp_path):
            raise FileNotFoundError(f"MLP checkpoint directory not found at {mlp_path}")
        self.gaussians.load_mlp_checkpoints(mlp_path)
        
        self.gaussians.eval()

        # 6. 设置背景
        bg_color = [1.0, 1.0, 1.0] if self.dataset.white_background else [0.0, 0.0, 0.0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 7. 加载cameras.json
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

        # 8. 加载cache.json
        if os.path.exists(cache_json_path):
            with open(cache_json_path, 'r', encoding='utf-8') as f:
                self.cache_json = json.load(f)
        else:
            raise FileNotFoundError(f"cache.json not found at {cache_json_path}")

        # 9. 加载regions_config.json
        if os.path.exists(regions_config_path):
            with open(regions_config_path, 'r', encoding='utf-8') as f:
                self.regions_config = json.load(f)
            
            # 构建camera_id_to_region映射
            for cam_id, cam_data in self.cache_json.items():
                if cam_id not in self.camera_id_to_region:
                    for region_idx, region in enumerate(self.regions_config):
                        region_name = region['name']
                        if cam_data['in_regions'].get(region_name, False) and (int(cam_id) < int(region['max_id'])):
                            self.camera_id_to_region[cam_id] = region_idx
                            break
        else:
            raise FileNotFoundError(f"regions_config.json not found at {regions_config_path}")

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
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        return R

    def find_most_similar_camera(self, position, rotation_matrix):
        """
        找到与给定相机位姿最相似的训练相机
        
        参数:
            position: 相机位置 [x, y, z]
            rotation_matrix: 相机旋转矩阵 (3x3)
        
        返回:
            最相似相机的信息字典
        """
        best_cam_id = None
        best_similarity = -float('inf')

        for cam_id, cam_data in self.cameras_json.items():
            # 计算位置相似度 (负欧氏距离)
            cam_pos = np.array(cam_data['position'])
            pos_dist = np.linalg.norm(np.array(position) - cam_pos)
            pos_similarity = -pos_dist

            # 计算旋转相似度 (使用矩阵迹)
            cam_rot = np.array(cam_data['rotation'])
            rot_trace = np.trace(np.dot(rotation_matrix.T, cam_rot))
            rot_similarity = (rot_trace - 1) / 2  # 归一化到 [-1, 1]

            # 组合相似度 (可以调整权重)
            total_similarity = pos_similarity + 0.5 * rot_similarity

            if total_similarity > best_similarity:
                best_similarity = total_similarity
                best_cam_id = cam_id

        return best_cam_id

    def render(self, position, quaternion, image_width=1920, image_height=1080, fovx=0.6911112070278503, fovy=0.5076887909988846):
        """
        从新的相机位姿渲染图像
        
        参数:
            position: 相机位置 [x, y, z] (全局坐标系)
            quaternion: 相机姿态四元数 [w, x, y, z]
            image_width: 图像宽度
            image_height: 图像高度
            fovx: 水平视场角
            fovy: 垂直视场角
        
        返回:
            渲染的图像张量
        """
        if self.gaussians is None:
            raise RuntimeError("Renderer not initialized. Call render_init first.")

        # 1. 四元数转旋转矩阵
        R = self.quaternion_to_rotation_matrix(quaternion)
        # 注意：根据现有代码的坐标系，可能需要调整旋转矩阵的方向
        R = R.T  # 转置以适配现有代码的预期

        # 2. 找到最相似的相机
        best_cam_id = self.find_most_similar_camera(position, R)
        camera_region = self.camera_id_to_region.get(str(best_cam_id), 0)
        ape_code = int(best_cam_id)
        print(f"📸 Most similar camera: ID={best_cam_id}, region={camera_region}, ape_code={ape_code}")

        # 3. 创建MiniCam对象
        T = np.array(position)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
        projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        
        view = MiniCam(
            width=image_width,
            height=image_height,
            fovy=fovy,
            fovx=fovx,
            znear=0.01,
            zfar=100.0,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform
        )

        # 4. 渲染
        with torch.no_grad():
            self.gaussians.set_anchor_mask(view.camera_center, self.iteration, 1.0)
            voxel_visible_mask = prefilter_voxel(view, self.gaussians, self.pipeline, self.background)
            render_pkg = render(view, self.gaussians, self.pipeline, self.background, visible_mask=voxel_visible_mask, ape_code=ape_code, camera_region=camera_region)
            rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        
        return rendering


# 使用示例
if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="Render new views from arbitrary camera poses")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument("-s", "--source_path", type=str, required=True, help="Path to source data directory")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to load")
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    args = parser.parse_args()
    
    # 初始化渲染器
    renderer = Renderer()
    
    # 构建路径
    model_path = args.model_path
    cameras_json_path = os.path.join(args.source_path, "cameras.json")
    cache_json_path = os.path.join(args.source_path, "cache.json")
    regions_config_path = os.path.join(args.source_path, "regions_config.json")
    
    print(f"📁 Model path: {model_path}")
    print(f"📁 Source path: {args.source_path}")
    print(f"📁 Cameras JSON: {cameras_json_path}")
    print(f"📁 Cache JSON: {cache_json_path}")
    print(f"📁 Regions config: {regions_config_path}")
    
    # 初始化
    renderer.render_init(
        model_path=model_path,
        cameras_json_path=cameras_json_path,
        cache_json_path=cache_json_path,
        regions_config_path=regions_config_path,
        iteration=args.iteration,
        white_background=args.white_background
    )
    
    # 示例：使用一个训练相机的位姿进行渲染测试
    test_cam_id = "0"
    test_cam = renderer.cameras_json[test_cam_id]
    test_position = test_cam['position']
    # 从旋转矩阵计算四元数（简化示例，实际使用时需要提供四元数）
    test_R = np.array(test_cam['rotation'])
    # 简单的旋转矩阵转四元数（仅用于示例，实际可以使用scipy等库）
    trace = np.trace(test_R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (test_R[2, 1] - test_R[1, 2]) / s
        qy = (test_R[0, 2] - test_R[2, 0]) / s
        qz = (test_R[1, 0] - test_R[0, 1]) / s
    elif (test_R[0, 0] > test_R[1, 1]) and (test_R[0, 0] > test_R[2, 2]):
        s = np.sqrt(1.0 + test_R[0, 0] - test_R[1, 1] - test_R[2, 2]) * 2
        qw = (test_R[2, 1] - test_R[1, 2]) / s
        qx = 0.25 * s
        qy = (test_R[0, 1] + test_R[1, 0]) / s
        qz = (test_R[0, 2] + test_R[2, 0]) / s
    elif test_R[1, 1] > test_R[2, 2]:
        s = np.sqrt(1.0 + test_R[1, 1] - test_R[0, 0] - test_R[2, 2]) * 2
        qw = (test_R[0, 2] - test_R[2, 0]) / s
        qx = (test_R[0, 1] + test_R[1, 0]) / s
        qy = 0.25 * s
        qz = (test_R[1, 2] + test_R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + test_R[2, 2] - test_R[0, 0] - test_R[1, 1]) * 2
        qw = (test_R[1, 0] - test_R[0, 1]) / s
        qx = (test_R[0, 2] + test_R[2, 0]) / s
        qy = (test_R[1, 2] + test_R[2, 1]) / s
        qz = 0.25 * s
    test_quaternion = [qw, qx, qy, qz]
    
    print(f"🔍 Testing with camera {test_cam_id}")
    print(f"   Position: {test_position}")
    print(f"   Quaternion: {test_quaternion}")
    
    # 渲染
    rendered_image = renderer.render(
        position=test_position,
        quaternion=test_quaternion
    )
    
    # 保存渲染结果
    import torchvision
    output_path = "test_render.png"
    torchvision.utils.save_image(rendered_image, output_path)
    print(f"✅ Rendered image saved to {output_path}")

