import os
import torch
import json
import numpy as np


class AnchorFeatureStorage:
    """
    用于存储高斯球的_anchor_feat属性，使用字典结构，键为(region, moment, position)
    支持增量式保存和加载，优化了大文件的加载和匹配性能
    
    核心优化：延迟加载，只在需要时才处理数据
    """
    def __init__(self, save_path, format='pt', lazy_load=True):
        """
        初始化存储
        参数:
            save_path: 保存路径
            format: 保存格式，'pt'或'json'，默认为'pt'
            lazy_load: 是否延迟加载（只在需要时处理数据）
        """
        self.save_path = save_path
        self.format = format.lower()
        self.lazy_load = lazy_load
        self.storage_dict = {}  # 完整存储字典（只在非延迟模式时填充）
        self.positions_dict = {}  # 按 (region, moment) 分组的位置 numpy 数组
        self.features_dict = {}  # 按 (region, moment) 分组的特征
        self.kdtrees = {}  # 按 (region, moment) 分组的 KDTree 索引
        self._group_index = None  # 快速分组索引（轻量级，只包含键）
        self._raw_loaded = False  # 是否已加载原始数据
        
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # 尝试加载已有的数据
        self._load_metadata()
    
    def _load_metadata(self):
        """只加载元数据（索引信息），不加载完整数据"""
        metadata_path = self.save_path.replace('.pt', '_metadata.pt').replace('.json', '_metadata.json')
        
        # 检查是否有元数据文件
        if os.path.exists(metadata_path):
            print(f"[AnchorFeatureStorage] Loading metadata from {metadata_path}")
            try:
                if metadata_path.endswith('.pt'):
                    metadata = torch.load(metadata_path)
                else:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                self._group_index = metadata.get('group_index', {})
                print(f"[AnchorFeatureStorage] Metadata loaded: {len(self._group_index)} groups")
                for cache_key_str, count in sorted(self._group_index.items()):
                    parts = cache_key_str.split('_')
                    region, moment = int(parts[0]), int(parts[1])
                    print(f"[AnchorFeatureStorage]   - Region {region}, Moment {moment}: {count} entries")
                
                return True
            except Exception as e:
                print(f"[AnchorFeatureStorage] Failed to load metadata: {e}")
                return False
        
        return False
    
    def _build_metadata(self):
        """构建元数据（索引信息）"""
        print(f"[AnchorFeatureStorage] Building metadata index...")
        import time
        start_time = time.time()
        
        # 快速遍历一次，构建轻量级索引
        self._group_index = {}
        for key in self.storage_dict.keys():
            region, moment, pos = key
            cache_key = f"{region}_{moment}"
            if cache_key not in self._group_index:
                self._group_index[cache_key] = 0
            self._group_index[cache_key] += 1
        
        build_time = time.time() - start_time
        print(f"[AnchorFeatureStorage] Metadata built in {build_time:.2f}s: {len(self._group_index)} groups")
        
        # 保存元数据
        metadata_path = self.save_path.replace('.pt', '_metadata.pt').replace('.json', '_metadata.json')
        try:
            if metadata_path.endswith('.pt'):
                torch.save({'group_index': self._group_index}, metadata_path)
            else:
                with open(metadata_path, 'w') as f:
                    json.dump({'group_index': self._group_index}, f)
            print(f"[AnchorFeatureStorage] Metadata saved to {metadata_path}")
        except Exception as e:
            print(f"[AnchorFeatureStorage] Failed to save metadata: {e}")
    
    def _load_data_for_key(self, cache_key):
        """加载特定 (region, moment) 的数据"""
        region, moment = cache_key
        
        if cache_key in self.positions_dict:
            return  # 已经加载过了
        
        print(f"[AnchorFeatureStorage] Loading data for region {region}, moment {moment}...")
        import time
        start_time = time.time()
        
        positions_list = []
        features_list = []
        
        # 如果已经加载了完整数据，直接筛选
        if self.storage_dict:
            for key, feat in self.storage_dict.items():
                key_region, key_moment, pos = key
                if key_region == region and key_moment == moment:
                    positions_list.append(pos)
                    features_list.append(feat)
        else:
            # 需要从文件加载
            print(f"[AnchorFeatureStorage] Storage dict is empty, loading from file...")
            self._load_full_data()
            
            for key, feat in self.storage_dict.items():
                key_region, key_moment, pos = key
                if key_region == region and key_moment == moment:
                    positions_list.append(pos)
                    features_list.append(feat)
        
        if len(positions_list) == 0:
            print(f"[AnchorFeatureStorage] No data found for region {region}, moment {moment}")
            return
        
        print(f"[AnchorFeatureStorage] Found {len(positions_list)} entries, building KDTree...")
        
        # 转换为 numpy 数组
        positions_array = np.array(positions_list, dtype=np.float32)
        self.positions_dict[cache_key] = positions_array
        self.features_dict[cache_key] = features_list
        
        # 构建 KDTree
        try:
            from scipy.spatial import KDTree
            print(f"[AnchorFeatureStorage] Building KDTree...")
            self.kdtrees[cache_key] = KDTree(positions_array)
            print(f"[AnchorFeatureStorage] KDTree built successfully")
        except Exception as e:
            self.kdtrees[cache_key] = None
            print(f"[AnchorFeatureStorage] KDTree build failed: {e}")
        
        load_time = time.time() - start_time
        print(f"[AnchorFeatureStorage] Data loaded in {load_time:.2f}s")
    
    def _load_full_data(self):
        """加载完整数据（只在需要时调用）"""
        if self._raw_loaded:
            return
        
        print(f"[AnchorFeatureStorage] Loading full data from {self.save_path}...")
        import time
        start_time = time.time()
        
        try:
            if self.format == 'pt':
                self.storage_dict = torch.load(self.save_path)
            elif self.format == 'json':
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                    self.storage_dict = {}
                    for key_str, value in data.items():
                        parts = key_str.split(',')
                        region = int(parts[0])
                        moment = int(parts[1])
                        position = tuple(float(x) for x in parts[2:])
                        self.storage_dict[(region, moment, position)] = value
            
            self._raw_loaded = True
            load_time = time.time() - start_time
            print(f"[AnchorFeatureStorage] Loaded {len(self.storage_dict)} entries in {load_time:.2f}s")
            
        except Exception as e:
            print(f"[AnchorFeatureStorage] Error loading full data: {e}")
            import traceback
            traceback.print_exc()
            self.storage_dict = {}
    
    def _get_key(self, region, moment, position):
        """将(region, moment, position)转换为可哈希的键"""
        if isinstance(position, torch.Tensor):
            position = position.detach().cpu().numpy()
        if isinstance(position, np.ndarray):
            position = tuple(float(x) for x in position.flatten())
        elif not isinstance(position, (list, tuple)):
            position = (float(position),)
        return (int(region), int(moment), tuple(position))
    
    def _encode_tensor(self, tensor):
        """将tensor编码为可存储的格式"""
        if self.format == 'json':
            return tensor.detach().cpu().numpy().tolist()
        else:
            return tensor.detach().cpu()
    
    def _decode_tensor(self, data, device='cpu'):
        """将存储的数据解码为tensor"""
        if self.format == 'json':
            return torch.tensor(data, dtype=torch.float32, device=device)
        else:
            if isinstance(data, torch.Tensor):
                return data.to(device)
            return torch.tensor(data, dtype=torch.float32, device=device)
    
    def add(self, region, moment, positions, anchor_feats):
        """添加单个或多个高斯球的_anchor_feat"""
        if isinstance(positions, torch.Tensor) and positions.dim() == 1:
            positions = positions.unsqueeze(0)
            anchor_feats = anchor_feats.unsqueeze(0)
        
        for i in range(positions.shape[0]):
            key = self._get_key(region, moment, positions[i])
            self.storage_dict[key] = self._encode_tensor(anchor_feats[i])
        
        # 清除该组的缓存
        cache_key = (region, moment)
        if cache_key in self.positions_dict:
            del self.positions_dict[cache_key]
        if cache_key in self.features_dict:
            del self.features_dict[cache_key]
        if cache_key in self.kdtrees:
            del self.kdtrees[cache_key]
    
    def get(self, region, moment, position):
        """获取单个高斯球的_anchor_feat"""
        key = self._get_key(region, moment, position)
        if key in self.storage_dict:
            return self._decode_tensor(self.storage_dict[key])
        return None
    
    def get_region_moment(self, region, moment):
        """获取指定region和moment下的所有数据"""
        positions = []
        feats = []
        for key, feat in self.storage_dict.items():
            if key[0] == region and key[1] == moment:
                positions.append(torch.tensor(key[2]))
                feats.append(self._decode_tensor(feat))
        return positions, feats
    
    def match_and_assign_features(self, region, moment, query_positions, tolerance=1e-3):
        """
        根据位置匹配并返回对应的 anchor features（GPU版本）
        参数:
            region: 区域索引
            moment: 时刻值
            query_positions: 需要匹配的位置张量，形状为[N, 3]
            tolerance: 匹配的容差（距离）
        返回:
            assigned_feats: 匹配到的特征张量，形状为[N, feat_dim]
            match_mask: 布尔张量，指示哪些位置匹配成功
        """
        cache_key = (region, moment)
        
        # 延迟加载：按需加载数据
        self._load_data_for_key(cache_key)
        
        if cache_key not in self.positions_dict:
            print(f"[AnchorFeatureStorage] No data found for region {region}, moment {moment}")
            return None, torch.zeros(query_positions.shape[0], dtype=torch.bool, device=query_positions.device if isinstance(query_positions, torch.Tensor) else 'cpu')
        
        stored_positions = self.positions_dict[cache_key]
        stored_features = self.features_dict[cache_key]
        
        print(f"[AnchorFeatureStorage] Matching {query_positions.shape[0]} queries against {len(stored_positions)} stored positions for region {region}, moment {moment}")
        
        # 将 query_positions 转换为 numpy
        if isinstance(query_positions, torch.Tensor):
            query_np = query_positions.detach().cpu().numpy()
            device = query_positions.device
        else:
            query_np = np.array(query_positions)
            device = 'cpu'
        
        # 优先使用 KDTree
        use_kdtree = False
        if cache_key in self.kdtrees and self.kdtrees[cache_key] is not None:
            try:
                kdtree = self.kdtrees[cache_key]
                min_distances, closest_indices = kdtree.query(query_np, k=1)
                use_kdtree = True
                print(f"[AnchorFeatureStorage] Using KDTree for matching")
            except Exception as e:
                print(f"[AnchorFeatureStorage] KDTree query failed: {e}, falling back to batch matching")
                use_kdtree = False
        
        # 如果没有 KDTree，使用分批暴力匹配
        if not use_kdtree:
            print(f"[AnchorFeatureStorage] Using batch matching (memory efficient)")
            min_distances = np.full(query_np.shape[0], np.inf, dtype=np.float32)
            closest_indices = np.full(query_np.shape[0], -1, dtype=np.int64)
            
            batch_size = 1000
            for batch_start in range(0, query_np.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, query_np.shape[0])
                query_batch = query_np[batch_start:batch_end]
                
                diff = query_batch[:, np.newaxis, :] - stored_positions[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diff ** 2, axis=-1))
                
                batch_min_dists = np.min(distances, axis=1)
                batch_min_indices = np.argmin(distances, axis=1)
                
                min_distances[batch_start:batch_end] = batch_min_dists
                closest_indices[batch_start:batch_end] = batch_min_indices
        
        # 确定匹配
        match_mask = min_distances <= tolerance
        match_count = np.sum(match_mask)
        print(f"[AnchorFeatureStorage] Matched {match_count}/{query_positions.shape[0]} points (tolerance={tolerance})")
        
        # 获取特征维度
        feat_dim = None
        for feat in stored_features:
            if self.format == 'json':
                feat_dim = len(feat)
            else:
                feat_dim = feat.shape[0] if hasattr(feat, 'shape') else len(feat)
            break
        
        if feat_dim is None:
            return None, torch.zeros(query_positions.shape[0], dtype=torch.bool, device=device)
        
        # 直接在 GPU 上构建结果（关键优化！）
        print(f"[AnchorFeatureStorage] Assigning features to GPU (device={device})...")
        assigned_feats_np = np.zeros((query_positions.shape[0], feat_dim), dtype=np.float32)
        
        for i in range(query_positions.shape[0]):
            if match_mask[i]:
                if self.format == 'json':
                    assigned_feats_np[i] = np.array(stored_features[closest_indices[i]])
                else:
                    feat = stored_features[closest_indices[i]]
                    if isinstance(feat, torch.Tensor):
                        assigned_feats_np[i] = feat.numpy()
                    else:
                        assigned_feats_np[i] = np.array(feat)
        
        # 转换为 torch tensor 并直接放到 GPU
        assigned_feats = torch.tensor(assigned_feats_np, dtype=torch.float32, device=device)
        match_mask_tensor = torch.tensor(match_mask, dtype=torch.bool, device=device)
        
        return assigned_feats, match_mask_tensor
    
    def get_all_regions(self):
        """获取所有存在的区域ID"""
        regions = set()
        for key_str in self._group_index.keys() if self._group_index else []:
            parts = key_str.split('_')
            regions.add(int(parts[0]))
        
        # 如果 group_index 为空且已加载完整数据
        if not regions and self.storage_dict:
            for key in self.storage_dict.keys():
                regions.add(key[0])
        
        return sorted(list(regions))
    
    def get_all_moments(self):
        """获取所有存在的moment值"""
        moments = set()
        for key_str in self._group_index.keys() if self._group_index else []:
            parts = key_str.split('_')
            moments.add(int(parts[1]))
        
        # 如果 group_index 为空且已加载完整数据
        if not moments and self.storage_dict:
            for key in self.storage_dict.keys():
                moments.add(key[1])
        
        return sorted(list(moments))
    
    def save(self):
        """保存存储数据"""
        if self.format == 'pt':
            torch.save(self.storage_dict, self.save_path)
            print(f"[AnchorFeatureStorage] Saved {len(self.storage_dict)} entries to {self.save_path}")
        elif self.format == 'json':
            json_dict = {}
            for key, value in self.storage_dict.items():
                region, moment, position = key
                key_str = f"{region},{moment},{','.join(str(x) for x in position)}"
                json_dict[key_str] = value
            with open(self.save_path, 'w') as f:
                json.dump(json_dict, f, indent=2)
            print(f"[AnchorFeatureStorage] Saved {len(self.storage_dict)} entries to {self.save_path}")
        
        # 同时保存元数据
        self._build_metadata()
    
    def clear(self):
        """清空存储"""
        self.storage_dict = {}
        self.positions_dict = {}
        self.features_dict = {}
        self.kdtrees = {}
        self._raw_loaded = False
    
    def __len__(self):
        return len(self.storage_dict)
