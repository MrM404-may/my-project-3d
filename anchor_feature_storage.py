import os
import torch
import json
import numpy as np


class AnchorFeatureStorage:
    """
    用于存储高斯球的_anchor_feat属性，使用字典结构，键为(region, moment, position)
    支持增量式保存和加载，优化了大文件的加载和匹配性能
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
        self.storage_dict = {}  # 完整存储字典
        self.positions_dict = {}  # 按 (region, moment) 分组的位置 numpy 数组
        self.features_dict = {}  # 按 (region, moment) 分组的特征
        self.kdtrees = {}  # 按 (region, moment) 分组的 KDTree 索引
        self._group_index = None  # 快速分组索引
        self._raw_loaded = False  # 是否已加载原始数据
        
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # 尝试加载已有的数据
        self._load_existing()
    
    def _build_cache_for_key(self, cache_key):
        """只为特定的 (region, moment) 构建缓存"""
        region, moment = cache_key
        
        print(f"[AnchorFeatureStorage] Building cache for region {region}, moment {moment}...")
        
        # 收集该组的所有数据
        positions_list = []
        features_list = []
        
        # 快速查找：先尝试用分组索引（如果已构建）
        if hasattr(self, '_group_index') and self._group_index is not None:
            if cache_key in self._group_index:
                items = self._group_index[cache_key]
                positions_list = [pos for pos, feat in items]
                features_list = [feat for pos, feat in items]
        else:
            # 没有分组索引，遍历所有键（第一次较慢，但之后会构建索引）
            for key, feat in self.storage_dict.items():
                key_region, key_moment, pos = key
                if key_region == region and key_moment == moment:
                    positions_list.append(pos)
                    features_list.append(feat)
        
        if len(positions_list) == 0:
            print(f"[AnchorFeatureStorage] No data found for region {region}, moment {moment}")
            return
        
        print(f"[AnchorFeatureStorage] Found {len(positions_list)} entries for region {region}, moment {moment}")
        
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
        except ImportError:
            self.kdtrees[cache_key] = None
            print(f"[AnchorFeatureStorage] scipy not available, using batch matching")
        except Exception as e:
            self.kdtrees[cache_key] = None
            print(f"[AnchorFeatureStorage] KDTree build failed: {e}, using batch matching")
    
    def _build_full_cache(self):
        """构建完整缓存（用于非延迟加载模式）"""
        print(f"[AnchorFeatureStorage] Building cache for all regions/moments...")
        
        # 先按 (region, moment) 分组
        groups = {}
        for key, feat in self.storage_dict.items():
            region, moment, pos = key
            cache_key = (region, moment)
            if cache_key not in groups:
                groups[cache_key] = []
            groups[cache_key].append((pos, feat))
        
        # 为每个组构建缓存
        for cache_key, items in groups.items():
            positions_list = [pos for pos, feat in items]
            features_list = [feat for pos, feat in items]
            
            positions_array = np.array(positions_list, dtype=np.float32)
            self.positions_dict[cache_key] = positions_array
            self.features_dict[cache_key] = features_list
            
            # 构建 KDTree
            try:
                from scipy.spatial import KDTree
                self.kdtrees[cache_key] = KDTree(positions_array)
            except ImportError:
                self.kdtrees[cache_key] = None
        
        print(f"[AnchorFeatureStorage] Cache built for {len(groups)} groups")
    
    def _get_key(self, region, moment, position):
        """
        将(region, moment, position)转换为可哈希的键
        """
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
    
    def _decode_tensor(self, data):
        """将存储的数据解码为tensor"""
        if self.format == 'json':
            return torch.tensor(data, dtype=torch.float32)
        else:
            return data
    
    def add(self, region, moment, positions, anchor_feats):
        """
        添加单个或多个高斯球的_anchor_feat
        参数:
            region: 区域索引
            moment: 时刻值（0, 1, 2, 3等）
            positions: 位置张量，形状为[N, 3]或单个位置
            anchor_feats: 对应的_anchor_feat张量，形状为[N, feat_dim]或单个特征
        """
        # 处理单个情况
        if isinstance(positions, torch.Tensor) and positions.dim() == 1:
            positions = positions.unsqueeze(0)
            anchor_feats = anchor_feats.unsqueeze(0)
        
        # 遍历添加
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
        """
        获取单个高斯球的_anchor_feat
        """
        key = self._get_key(region, moment, position)
        if key in self.storage_dict:
            return self._decode_tensor(self.storage_dict[key])
        return None
    
    def get_region_moment(self, region, moment):
        """
        获取指定region和moment下的所有数据
        返回: (positions_list, feats_list)
        """
        positions = []
        feats = []
        for key, feat in self.storage_dict.items():
            if key[0] == region and key[1] == moment:
                positions.append(torch.tensor(key[2]))
                feats.append(self._decode_tensor(feat))
        return positions, feats
    
    def match_and_assign_features(self, region, moment, query_positions, tolerance=1e-3):
        """
        根据位置匹配并返回对应的 anchor features
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
        
        # 如果是延迟加载模式，且该组未缓存，先构建缓存
        if self.lazy_load and cache_key not in self.positions_dict and self._raw_loaded:
            self._build_cache_for_key(cache_key)
        
        if cache_key not in self.positions_dict:
            # 没有该 region 和 moment 的数据
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
        
        # 优先尝试 KDTree
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
        
        # 如果没有 KDTree 或者 KDTree 失败，使用分批暴力匹配
        if not use_kdtree:
            print(f"[AnchorFeatureStorage] Using batch matching (memory efficient)")
            min_distances = np.full(query_np.shape[0], np.inf, dtype=np.float32)
            closest_indices = np.full(query_np.shape[0], -1, dtype=np.int64)
            
            # 分批处理，避免内存爆炸
            batch_size = 1000  # 每批处理1000个查询点
            for batch_start in range(0, query_np.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, query_np.shape[0])
                query_batch = query_np[batch_start:batch_end]
                
                # 计算这个批次的距离
                # 使用更高效的方式：利用广播但限制批次大小
                # (B, 3) 和 (M, 3) -> (B, M)
                diff = query_batch[:, np.newaxis, :] - stored_positions[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diff ** 2, axis=-1))
                
                # 找最小距离
                batch_min_dists = np.min(distances, axis=1)
                batch_min_indices = np.argmin(distances, axis=1)
                
                min_distances[batch_start:batch_end] = batch_min_dists
                closest_indices[batch_start:batch_end] = batch_min_indices
        
        # 确定哪些匹配在容差范围内
        match_mask = min_distances <= tolerance
        match_count = np.sum(match_mask)
        print(f"[AnchorFeatureStorage] Matched {match_count}/{query_positions.shape[0]} points (tolerance={tolerance})")
        
        # 准备输出特征
        feat_dim = None
        for feat in stored_features:
            if self.format == 'json':
                feat_dim = len(feat)
            else:
                feat_dim = feat.shape[0] if hasattr(feat, 'shape') else len(feat)
            break
        
        if feat_dim is None:
            return None, torch.zeros(query_positions.shape[0], dtype=torch.bool, device=device)
        
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
        
        # 转换为 torch tensor
        assigned_feats = torch.tensor(assigned_feats_np, dtype=torch.float32, device=device)
        match_mask_tensor = torch.tensor(match_mask, dtype=torch.bool, device=device)
        
        return assigned_feats, match_mask_tensor
    
    def _load_existing(self):
        """加载已有的存储数据"""
        if not os.path.exists(self.save_path):
            print(f"[AnchorFeatureStorage] No existing file found at {self.save_path}, starting fresh")
            return
        
        try:
            if self.format == 'pt':
                import time
                print(f"[AnchorFeatureStorage] Loading data from {self.save_path}...")
                start_time = time.time()
                self.storage_dict = torch.load(self.save_path)
                load_time = time.time() - start_time
                print(f"[AnchorFeatureStorage] Loaded {len(self.storage_dict)} entries in {load_time:.2f}s")
            elif self.format == 'json':
                with open(self.save_path, 'r') as f:
                    # JSON中键是字符串，需要转换
                    data = json.load(f)
                    self.storage_dict = {}
                    for key_str, value in data.items():
                        # 解析键："region,moment,x,y,z" -> (region, moment, (x,y,z))
                        parts = key_str.split(',')
                        region = int(parts[0])
                        moment = int(parts[1])
                        position = tuple(float(x) for x in parts[2:])
                        self.storage_dict[(region, moment, position)] = value
                print(f"[AnchorFeatureStorage] Loaded {len(self.storage_dict)} entries from {self.save_path}")
            
            self._raw_loaded = True
            
            # 构建分组索引（一次性遍历，之后快速查找）
            try:
                print(f"[AnchorFeatureStorage] Building group index...")
                import time
                start_time = time.time()
                
                self._group_index = {}
                for key, feat in self.storage_dict.items():
                    region, moment, pos = key
                    cache_key = (region, moment)
                    if cache_key not in self._group_index:
                        self._group_index[cache_key] = []
                    self._group_index[cache_key].append((pos, feat))
                
                build_time = time.time() - start_time
                print(f"[AnchorFeatureStorage] Group index built in {build_time:.2f}s: {len(self._group_index)} groups")
                for cache_key in sorted(self._group_index.keys()):
                    print(f"[AnchorFeatureStorage]   - Region {cache_key[0]}, Moment {cache_key[1]}: {len(self._group_index[cache_key])} entries")
            except Exception as e:
                print(f"[AnchorFeatureStorage] Failed to build group index: {e}")
                self._group_index = None
            
            # 如果不是延迟加载模式，构建完整缓存
            if not self.lazy_load:
                self._build_full_cache()
            else:
                print(f"[AnchorFeatureStorage] Lazy load enabled - cache will be built on demand")
        except Exception as e:
            print(f"[AnchorFeatureStorage] Error loading existing file: {e}, starting fresh")
            import traceback
            traceback.print_exc()
            self.storage_dict = {}
            self._group_index = None
    
    def save(self):
        """保存存储数据"""
        if self.format == 'pt':
            torch.save(self.storage_dict, self.save_path)
            print(f"[AnchorFeatureStorage] Saved {len(self.storage_dict)} entries to {self.save_path}")
        elif self.format == 'json':
            # 将键转换为字符串
            json_dict = {}
            for key, value in self.storage_dict.items():
                region, moment, position = key
                key_str = f"{region},{moment},{','.join(str(x) for x in position)}"
                json_dict[key_str] = value
            with open(self.save_path, 'w') as f:
                json.dump(json_dict, f, indent=2)
            print(f"[AnchorFeatureStorage] Saved {len(self.storage_dict)} entries to {self.save_path}")
    
    def get_all_regions(self):
        """获取所有存在的区域ID"""
        regions = set()
        for key in self.storage_dict.keys():
            regions.add(key[0])
        return sorted(list(regions))
    
    def get_all_moments(self):
        """获取所有存在的moment值"""
        moments = set()
        for key in self.storage_dict.keys():
            moments.add(key[1])
        return sorted(list(moments))
    
    def clear(self):
        """清空存储"""
        self.storage_dict = {}
        self.positions_dict = {}
        self.features_dict = {}
        self.kdtrees = {}
        self._raw_loaded = False
    
    def __len__(self):
        return len(self.storage_dict)
