import os
import torch
import json
import numpy as np
from plyfile import PlyData, PlyElement


class AnchorFeatureStoragePLY:
    """
    基于 PLY 格式的 anchor feature 存储
    
    核心优势：
    - 加载速度快（二进制数组）
    - 内存效率高（连续存储）
    - 按 region/moment 分组
    """
    def __init__(self, base_path, lazy_load=True):
        """
        初始化存储
        参数:
            base_path: 基础路径，会自动生成索引和数据文件
            lazy_load: 是否延迟加载
        """
        self.base_path = base_path
        self.lazy_load = lazy_load
        self.index_path = f"{base_path}_index.json"
        self.data_dir = f"{base_path}_data"
        
        # 数据缓存
        self._cached_data = {}  # {(region, moment): (positions, features)}
        self._loaded_groups = set()
        
        # 创建目录
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 加载索引
        self._load_index()
    
    def _load_index(self):
        """加载索引文件"""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'r') as f:
                    self.index = json.load(f)
                print(f"[AnchorFeatureStoragePLY] Loaded index: {len(self.index)} groups")
                for key_str in sorted(self.index.keys()):
                    info = self.index[key_str]
                    print(f"[AnchorFeatureStoragePLY]   - {key_str}: {info['count']} entries")
            except Exception as e:
                print(f"[AnchorFeatureStoragePLY] Failed to load index: {e}")
                self.index = {}
        else:
            self.index = {}
    
    def _save_index(self):
        """保存索引文件"""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
        print(f"[AnchorFeatureStoragePLY] Index saved to {self.index_path}")
    
    def _get_group_file(self, region, moment):
        """获取 region/moment 对应的 PLY 文件路径"""
        return os.path.join(self.data_dir, f"region{region}_moment{moment}.ply")
    
    def _save_group(self, region, moment, positions, features):
        """保存单个 group 到 PLY"""
        print(f"[AnchorFeatureStoragePLY] Saving group: region={region}, moment={moment}")
        
        # 准备数据
        positions_np = np.array(positions, dtype=np.float32) if positions else np.array([])
        features_np = np.array(features, dtype=np.float32) if features else np.array([])
        
        # 构建 PLY 数据结构
        vertex_data = []
        num_features = features_np.shape[1] if len(features_np) > 0 else 0
        
        # 准备数据类型定义
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        for i in range(num_features):
            dtype.append((f'f_{i}', 'f4'))
        
        # 填充数据
        if len(positions_np) > 0:
            vertex_arr = np.zeros(positions_np.shape[0], dtype=dtype)
            vertex_arr['x'] = positions_np[:, 0]
            vertex_arr['y'] = positions_np[:, 1]
            vertex_arr['z'] = positions_np[:, 2]
            
            for i in range(num_features):
                vertex_arr[f'f_{i}'] = features_np[:, i]
            
            el = PlyElement.describe(vertex_arr, 'vertex')
            ply_file = self._get_group_file(region, moment)
            PlyData([el]).write(ply_file)
            print(f"[AnchorFeatureStoragePLY] Saved {positions_np.shape[0]} entries to {ply_file}")
        
        # 更新索引
        key_str = f"{region}_{moment}"
        self.index[key_str] = {
            'count': len(positions_np) if len(positions_np) > 0 else 0,
            'feature_dim': num_features
        }
        self._save_index()
    
    def _load_group(self, region, moment):
        """加载单个 group 的 PLY 数据"""
        key_str = f"{region}_{moment}"
        
        if (region, moment) in self._cached_data:
            return self._cached_data[(region, moment)]
        
        ply_file = self._get_group_file(region, moment)
        
        if not os.path.exists(ply_file):
            print(f"[AnchorFeatureStoragePLY] File not found: {ply_file}")
            return None
        
        print(f"[AnchorFeatureStoragePLY] Loading from {ply_file}")
        import time
        start_time = time.time()
        
        try:
            plydata = PlyData.read(ply_file)
            
            # 读取位置
            positions = np.stack([
                np.asarray(plydata.elements[0]['x']),
                np.asarray(plydata.elements[0]['y']),
                np.asarray(plydata.elements[0]['z'])
            ], axis=1).astype(np.float32)
            
            # 读取特征
            feature_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('f_')]
            feature_names.sort(key=lambda x: int(x.split('_')[1]))
            
            if feature_names:
                features = np.zeros((positions.shape[0], len(feature_names)), dtype=np.float32)
                for i, name in enumerate(feature_names):
                    features[:, i] = np.asarray(plydata.elements[0][name]).astype(np.float32)
            else:
                features = np.array([])
            
            load_time = time.time() - start_time
            print(f"[AnchorFeatureStoragePLY] Loaded {positions.shape[0]} entries in {load_time:.2f}s")
            
            self._cached_data[(region, moment)] = (positions, features)
            return (positions, features)
            
        except Exception as e:
            print(f"[AnchorFeatureStoragePLY] Failed to load {ply_file}: {e}")
            return None
    
    def add(self, region, moment, positions, features):
        """添加数据（增量保存）"""
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()
        
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        
        # 如果已有数据，先加载旧数据
        key_str = f"{region}_{moment}"
        if key_str in self.index:
            old_data = self._load_group(region, moment)
            if old_data is not None:
                old_pos, old_feat = old_data
                if len(old_pos) > 0:
                    positions = np.vstack([old_pos, positions])
                    features = np.vstack([old_feat, features])
        
        # 保存
        self._save_group(region, moment, positions, features)
        
        # 更新缓存
        self._cached_data[(region, moment)] = (positions, features)
    
    def match_and_assign_features(self, region, moment, query_positions, tolerance=1e-3):
        """匹配特征"""
        print(f"[AnchorFeatureStoragePLY] Matching for region={region}, moment={moment}")
        
        data = self._load_group(region, moment)
        
        if data is None:
            print(f"[AnchorFeatureStoragePLY] No data found for region={region}, moment={moment}")
            if isinstance(query_positions, torch.Tensor):
                return None, torch.zeros(query_positions.shape[0], dtype=torch.bool, device=query_positions.device)
            else:
                return None, torch.zeros(len(query_positions), dtype=torch.bool)
        
        stored_positions, stored_features = data
        
        print(f"[AnchorFeatureStoragePLY] Matching {query_positions.shape[0]} queries against {stored_positions.shape[0]} stored positions")
        
        # 将 query_positions 转换为 numpy
        if isinstance(query_positions, torch.Tensor):
            query_np = query_positions.detach().cpu().numpy()
            device = query_positions.device
        else:
            query_np = np.array(query_positions)
            device = 'cpu'
        
        # 使用 KDTree
        try:
            from scipy.spatial import KDTree
            print(f"[AnchorFeatureStoragePLY] Building KDTree...")
            kdtree = KDTree(stored_positions)
            min_distances, closest_indices = kdtree.query(query_np, k=1)
            print(f"[AnchorFeatureStoragePLY] KDTree query complete")
        except Exception as e:
            print(f"[AnchorFeatureStoragePLY] KDTree failed: {e}, using batch matching")
            # 回退到分批匹配
            min_distances = np.full(query_np.shape[0], np.inf, dtype=np.float32)
            closest_indices = np.full(query_np.shape[0], -1, dtype=np.int64)
            
            batch_size = 1000
            for batch_start in range(0, query_np.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, query_np.shape[0])
                query_batch = query_np[batch_start:batch_end]
                
                diff = query_batch[:, np.newaxis, :] - stored_positions[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diff ** 2, axis=-1))
                
                min_distances[batch_start:batch_end] = np.min(distances, axis=1)
                closest_indices[batch_start:batch_end] = np.argmin(distances, axis=1)
        
        match_mask = min_distances <= tolerance
        match_count = np.sum(match_mask)
        print(f"[AnchorFeatureStoragePLY] Matched {match_count}/{query_positions.shape[0]} points")
        
        # 准备结果
        if stored_features.size > 0:
            assigned_feats_np = np.zeros((query_positions.shape[0], stored_features.shape[1]), dtype=np.float32)
            for i in range(query_positions.shape[0]):
                if match_mask[i]:
                    assigned_feats_np[i] = stored_features[closest_indices[i]]
            
            assigned_feats = torch.tensor(assigned_feats_np, dtype=torch.float32, device=device)
        else:
            assigned_feats = None
        
        match_mask_tensor = torch.tensor(match_mask, dtype=torch.bool, device=device)
        
        return assigned_feats, match_mask_tensor
    
    def get_all_regions(self):
        """获取所有 region"""
        regions = set()
        for key_str in self.index.keys():
            region, moment = map(int, key_str.split('_'))
            regions.add(region)
        return sorted(list(regions))
    
    def get_all_moments(self):
        """获取所有 moment"""
        moments = set()
        for key_str in self.index.keys():
            region, moment = map(int, key_str.split('_'))
            moments.add(moment)
        return sorted(list(moments))
    
    def __len__(self):
        """总条目数"""
        total = 0
        for key_str, info in self.index.items():
            total += info['count']
        return total
