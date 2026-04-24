import json
import os
import cv2
import numpy as np
from matplotlib.path import Path

# ====================== 内部辅助类（对外不可见） ======================
class _CameraRegionInfo:
    """内部类：存储单个相机的区域相关信息（预计算复用）"""
    def __init__(self, camera_id):
        self.id = camera_id
        self.position = None
        self.rotation = None
        self.img_name = None
        self.in_regions = {}  # {区域名: 是否在该区域位置内}
        self.mask_ratios = {} # {区域名: 该区域的掩码占比}

# ====================== 基础工具函数（内部复用） ======================
def _load_camera_data(json_file_path):
    """加载JSON文件中的相机数据（内部函数）"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            if os.path.getsize(json_file_path) > 0:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
                return data
            else:
                print(f"警告: {json_file_path} 文件为空")
                return []
    except json.JSONDecodeError:
        camera_data = []
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        camera_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"警告: 第 {line_num+1} 行JSON解析失败: {e}")
                        continue
        return camera_data
    except Exception as e:
        print(f"读取文件 {json_file_path} 时出错: {e}")
        return []

def _is_point_in_polygon(points, polygon):
    """判断点是否在多边形内（内部函数）"""
    path = Path(polygon)
    return path.contains_points(points)

def _calculate_mask_ratio(mask_img):
    """计算掩码中有效区域(255)的占比（内部函数）"""
    total_pixels = mask_img.shape[0] * mask_img.shape[1]
    if total_pixels == 0: return 0.0
    valid_pixels = np.sum(mask_img == 255)
    return valid_pixels / total_pixels

def _generate_region_mask(depth_img_path, camera_pose, region_vertices, camera_intrinsics, is_camera_raw=False):
    """生成区域掩码图（内部函数）"""
    fx, fy, cx, cy = camera_intrinsics
    
    # 1. 读取深度图
    depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        return None
    
    if len(depth_img.shape) == 3:
        depth_img_original = depth_img[:, :, 0]
    else:
        depth_img_original = depth_img
    
    h, w = depth_img_original.shape

    # 2. 深度处理
    norm_max_depth = 20.0
    norm_min_depth = 0.0
    invert_depth = True
    flip_y = True
    flip_x = False

    if is_camera_raw:
        depth_scale = 100.0
        depth_meter = depth_img_original.astype(np.float32) / depth_scale
        if invert_depth:
            depth_meter = norm_max_depth - depth_meter
            depth_meter = np.clip(depth_meter, 0.01, norm_max_depth)
    else:
        if depth_img_original.max() > 0:
            alpha = 255.0 / depth_img_original.max()
            depth_normalized = cv2.convertScaleAbs(depth_img_original, alpha=alpha)
        else:
            depth_normalized = depth_img_original.astype(np.uint8)
            
        depth_ratio = 1 - depth_normalized.astype(np.float32) / 255.0
        depth_meter = depth_ratio * (norm_max_depth - norm_min_depth) + norm_min_depth
        if invert_depth:
            depth_meter = norm_max_depth - depth_meter
            depth_meter = np.clip(depth_meter, 0.01, norm_max_depth)

    # 3. 生成点云
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    u = u_coords.reshape(-1)
    v = v_coords.reshape(-1)
    depth_flat = depth_meter.reshape(-1)

    valid_mask = depth_flat > 0
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    d_valid = depth_flat[valid_mask]

    x = (u_valid - cx) * d_valid / fx
    if flip_x: x = -x
    y = (v_valid - cy) * d_valid / fy
    if flip_y: y = -y
    z = d_valid

    point_cloud_cam = np.column_stack((x, y, z))
    
    if camera_pose is not None:
        R = np.array(camera_pose["rotation"], dtype=np.float32)
        T = np.array(camera_pose["position"], dtype=np.float32)
        point_cloud_world = np.dot(R, point_cloud_cam.T).T + T
    else:
        point_cloud_world = point_cloud_cam

    # 4. 生成掩码
    mask_img = np.zeros((h * w,), dtype=np.uint8)
    if len(point_cloud_world) > 0:
        path = Path(region_vertices)
        in_region = path.contains_points(point_cloud_world[:, [0, 2]])
        mask_img[valid_mask] = in_region * 255
    
    return mask_img.reshape((h, w))

# ====================== 缓存相关工具函数 ======================
def _save_camera_cache(camera_info_dict, cache_path, verbose=True):
    """保存相机预计算结果到缓存文件"""
    try:
        # 转换numpy类型为原生Python类型（JSON序列化需要）
        serializable_dict = {}
        for cam_id, cam_data in camera_info_dict.items():
            serializable_data = cam_data.copy()
            # 转换in_regions中的numpy bool为Python bool
            serializable_data['in_regions'] = {k: bool(v) for k, v in cam_data['in_regions'].items()}
            # 转换mask_ratios中的numpy float为Python float
            serializable_data['mask_ratios'] = {k: float(v) for k, v in cam_data['mask_ratios'].items()}
            serializable_dict[cam_id] = serializable_data
        
        # 保存到JSON文件
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_dict, f, ensure_ascii=False, indent=2)
        
        if verbose:
            print(f"✅ 预计算缓存已保存到: {cache_path}")
        return True
    except Exception as e:
        if verbose:
            print(f"❌ 保存缓存失败: {e}")
        return False

def _load_camera_cache(cache_path, regions_config, verbose=True):
    """从缓存文件加载相机预计算结果"""
    if not os.path.exists(cache_path):
        if verbose:
            print(f"ℹ️  缓存文件不存在: {cache_path}")
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # 验证缓存数据完整性（匹配当前区域配置）
        region_names = [r['name'] for r in regions_config]
        for cam_id, cam_data in cache_data.items():
            # 检查必要字段
            required_fields = ['id', 'position', 'rotation', 'img_name', 'in_regions', 'mask_ratios']
            if not all(f in cam_data for f in required_fields):
                if verbose:
                    print(f"⚠️  相机 {cam_id} 缓存数据不完整，跳过缓存")
                return None
            
            # 检查区域匹配
            cache_region_names = list(cam_data['in_regions'].keys())
            if sorted(cache_region_names) != sorted(region_names):
                if verbose:
                    print(f"⚠️  缓存区域配置不匹配（缓存: {cache_region_names}, 当前: {region_names}），跳过缓存")
                return None
        
        if verbose:
            print(f"✅ 成功加载缓存文件: {cache_path} (共 {len(cache_data)} 个相机)")
        return cache_data
    except Exception as e:
        if verbose:
            print(f"❌ 加载缓存失败: {e}")
        return None

# ====================== 优化后的核心函数（新增缓存功能） ======================
def get_camera_ids_by_regions(
    json_path, 
    regions_config, 
    depth_images_dir="", 
    camera_intrinsics=None,
    mask_ratio_threshold=0.0,
    depth_img_ext='.png',
    is_camera_raw=False,
    verbose=True,
    cache_json_path=None  # 新增：缓存文件路径
):
    """
    【兼容原调用方式 + 缓存优化 + 内部性能优化】
    增强版筛选函数：
    1. 如果相机自身在区域内 -> 入选
    2. 如果相机自身不在区域内，但看到该区域的占比 >= mask_ratio_threshold -> 入选
    3. 新增缓存功能：复用预计算结果，避免重复耗时计算
    
    参数:
        json_path: 相机数据 JSON 文件路径
        regions_config: 区域配置列表 [{'name': 'R1', 'vertices': [[...]]}, ...]
        depth_images_dir: 深度图存放目录 (如果使用掩码筛选，此项必填)
        camera_intrinsics: 相机内参元组 (fx, fy, cx, cy)
        mask_ratio_threshold: 掩码占比阈值 (0.0 ~ 1.0)，例如 0.3 代表 30%
        depth_img_ext: 深度图文件后缀
        is_camera_raw: 深度图是否为Raw格式
        verbose: 是否打印详细日志
        cache_json_path: 缓存文件路径（None则不使用缓存，否则读取/保存缓存）
    
    返回:
        all_region_ids: 列表，长度等于 regions_config，每个元素是该区域的 ID 列表
        camera_info_dict: 所有相机的详细信息字典
    """
    # ---------------------- 步骤0：尝试加载缓存（新增核心逻辑） ----------------------
    camera_info_dict = None
    if cache_json_path is not None:
        camera_info_dict = _load_camera_cache(cache_json_path, regions_config, verbose)
    
    # ---------------------- 步骤1：无缓存时执行预计算（原有逻辑） ----------------------
    if camera_info_dict is None:
        # 1.1 加载原始相机数据
        all_camera_data = []
        if os.path.isdir(json_path):
            for filename in os.listdir(json_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(json_path, filename)
                    all_camera_data.extend(_load_camera_data(file_path))
        elif os.path.isfile(json_path):
            all_camera_data = _load_camera_data(json_path)
        
        # 1.2 初始化相机信息类（预计算位置归属 + 掩码占比）
        camera_info_list = []
        camera_info_dict = {}  # 最终返回的相机信息字典
        for camera in all_camera_data:
            if not ('id' in camera and 'position' in camera and len(camera['position']) >= 3):
                continue
            
            # 初始化内部相机信息类
            cam_info = _CameraRegionInfo(camera['id'])
            cam_info.position = camera['position']
            cam_info.rotation = camera.get('rotation')
            cam_info.img_name = camera.get('img_name', f"{cam_info.id}.jpg")
            
            # 预计算：位置归属（所有区域）
            xz_pos = [cam_info.position[0], cam_info.position[2]]
            for region in regions_config:
                region_name = region['name']
                vertices = np.array(region['vertices'])
                cam_info.in_regions[region_name] = _is_point_in_polygon([xz_pos], vertices)[0]
            
            # 预计算：掩码占比（所有区域，仅当条件满足时）
            if depth_images_dir and camera_intrinsics is not None:
                # 寻找深度图文件
                base_fn = os.path.splitext(cam_info.img_name)[0]
                depth_path = os.path.join(depth_images_dir, base_fn + depth_img_ext)
                if not os.path.exists(depth_path):
                    depth_path = os.path.join(depth_images_dir, cam_info.img_name)
                
                if os.path.exists(depth_path):
                    cam_pose = {
                        "position": cam_info.position,
                        "rotation": cam_info.rotation
                    }
                    # 一次性计算该相机对所有区域的掩码占比（避免重复计算）
                    for region in regions_config:
                        region_name = region['name']
                        vertices = np.array(region['vertices'])
                        try:
                            mask = _generate_region_mask(
                                depth_path, cam_pose, vertices, camera_intrinsics, is_camera_raw
                            )
                            cam_info.mask_ratios[region_name] = _calculate_mask_ratio(mask) if mask is not None else 0.0
                        except Exception as e:
                            cam_info.mask_ratios[region_name] = 0.0
                            if verbose:
                                print(f"警告: 相机 {cam_info.id} 计算区域 {region_name} 掩码失败: {e}")
                else:
                    # 无深度图，掩码占比设为0
                    for region in regions_config:
                        cam_info.mask_ratios[region['name']] = 0.0
            else:
                # 无深度图/内参，掩码占比设为0
                for region in regions_config:
                    cam_info.mask_ratios[region['name']] = 0.0
            
            # 加入列表和字典
            camera_info_list.append(cam_info)
            camera_info_dict[cam_info.id] = {
                'id': cam_info.id,
                'position': cam_info.position,
                'rotation': cam_info.rotation,
                'img_name': cam_info.img_name,
                'in_regions': cam_info.in_regions,  # 新增：位置归属信息
                'mask_ratios': cam_info.mask_ratios # 新增：掩码占比信息
            }
        
        # 1.3 保存缓存（如果指定了缓存路径）
        if cache_json_path is not None:
            _save_camera_cache(camera_info_dict, cache_json_path, verbose)
    
    # ---------------------- 步骤2：构建camera_info_list（适配缓存/预计算两种场景） ----------------------
    camera_info_list = []
    for cam_id, cam_data in camera_info_dict.items():
        cam_info = _CameraRegionInfo(cam_id)
        cam_info.id = cam_data['id']
        cam_info.position = cam_data['position']
        cam_info.rotation = cam_data['rotation']
        cam_info.img_name = cam_data['img_name']
        cam_info.in_regions = cam_data['in_regions']
        cam_info.mask_ratios = cam_data['mask_ratios']
        camera_info_list.append(cam_info)
    
    if verbose and not camera_info_list:
        print("警告: 没有有效的相机数据")
        return [[] for _ in regions_config], camera_info_dict
    
    # ---------------------- 步骤3：基于预处理结果筛选（无重复计算） ----------------------
    all_region_ids = []
    for region in regions_config:
        region_name = region['name']
        selected_ids = []
        count_pos = 0
        count_mask = 0
        
        for cam_info in camera_info_list:
            # 规则1: 自身在区域内，直接入选
            if cam_info.in_regions.get(region_name, False):
                selected_ids.append(cam_info.id)
                count_pos += 1
                continue
            
            # 规则2: 不在区域内，但掩码占比达标
            if mask_ratio_threshold > 0:
                ratio = cam_info.mask_ratios.get(region_name, 0.0)
                if ratio >= mask_ratio_threshold:
                    selected_ids.append(cam_info.id)
                    count_mask += 1
                    if verbose:
                        print(f"  相机 {cam_info.id} 入选 (掩码占比: {ratio:.2%})")
        
        all_region_ids.append(selected_ids)
        if verbose:
            print(f"\n区域 {region_name} 筛选完成: 共 {len(selected_ids)} 个 (位置: {count_pos}, 掩码: {count_mask})")

    return all_region_ids, camera_info_dict

# ====================== 调用示例（新增缓存参数） ======================
if __name__ == "__main__":
    # 1. 配置路径
    json_file_path = r"C:\Users\M\Desktop\block\macameras.json"
    depth_dir = r"C:\Users\M\Desktop\block" # 深度图目录
    cache_path = r"C:\Users\M\Desktop\block\camera_cache.json"  # 新增：缓存文件路径
    
    # 2. 配置区域
    regions_config = [
        {
            "name": "区域1",
            "vertices": [
                [-38.9, 83.7],
                [-33.1, -8.5],
                [-5.3, -12.4],
                [-11.8, 71.2]
            ]
        },
        {
            "name": "区域2", 
            "vertices": [
                [-14.0, 73.0],
                [-7.5, -15.6],
                [14.7, -14.0],
                [4.7, 72.0]
            ]
        }
    ]
    
    # 3. 配置相机内参
    MY_CAMERA_INTRINSICS = (
        867.3526294468791,  # fx
        861.2148494110500,  # fy
        650.3807935145040,  # cx
        364.2087458587477   # cy
    )
    
    # 4. 调用函数（新增cache_json_path参数）
    all_region_ids, cam_info_dict = get_camera_ids_by_regions(
        json_path=json_file_path,
        regions_config=regions_config,
        depth_images_dir=depth_dir,        
        camera_intrinsics=MY_CAMERA_INTRINSICS,
        mask_ratio_threshold=0.3,           
        depth_img_ext='.png',               
        is_camera_raw=False,                
        verbose=True,
        cache_json_path=cache_path  # 新增：指定缓存文件路径
    )
    
    # 5. 输出结果
    for i, ids in enumerate(all_region_ids):
        print(f"\n区域 {i+1} 最终 ID 列表 (数量: {len(ids)}):")
        print(ids)