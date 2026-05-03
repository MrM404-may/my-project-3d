class RenderingAnchorFeatStorage:
    """
    渲染用的特征存储类
    
    核心设计：用高斯球的位置 (x, y, z) 作为唯一标识符
    这样即使合并多个区域，也能通过高斯球位置精确定位到对应的特征
    
    存储结构设计：
        - {(region, moment): {anchor_position_tuple: feat_tensor}}
        - anchor_position_tuple: (x, y, z) 元组，精度到小数点后6位
    """
    def __init__(self, feat_dim: int, device='cuda', precision: int = 6):
        self.feat_dim = feat_dim
        self.device = device
        self.precision = precision
        # 存储结构：{(region, moment): {anchor_position_tuple: feat_tensor}}
        self._storage = {}
        # 辅助缓存：{(region, moment): {'poses': [list of tuples], 'feats': Tensor}}
        self._tensor_cache = {}
    
    def _position_to_key(self, position):
        """
        将高斯球位置转换为可哈希的键
        Args:
            position: 张量 [3] 或列表 [x, y, z]
        Returns:
            tuple: (x_rounded, y_rounded, z_rounded)
        """
        if isinstance(position, torch.Tensor):
            pos_list = position.cpu().tolist()
        else:
            pos_list = list(position)
        
        return tuple(round(p, self.precision) for p in pos_list)
    
    def _tensor_to_keys(self, positions):
        """
        将高斯球位置张量批量转换为键列表
        Args:
            positions: 张量 [N, 3]
        Returns:
            list of tuple
        """
        keys = []
        for pos in positions:
            keys.append(self._position_to_key(pos))
        return keys
    
    def add_feature(self, region: int, moment: int, anchor_position, feat: torch.Tensor):
        """
        添加单个高斯的特征
        Args:
            region: 区域标识
            moment: 时刻标识
            anchor_position: 高斯球位置，张量 [3] 或元组/列表 [x, y, z]
            feat: 特征张量 [feat_dim]
        """
        key = (region, moment)
        if key not in self._storage:
            self._storage[key] = {}
        
        position_key = self._position_to_key(anchor_position)
        self._storage[key][position_key] = feat
        
        # 清除缓存
        if key in self._tensor_cache:
            del self._tensor_cache[key]
    
    def add_region_features_batch(self, region: int, moment: int, 
                                   anchor_positions, feats: torch.Tensor):
        """
        批量添加某个区域的特征
        Args:
            region: 区域标识
            moment: 时刻标识
            anchor_positions: 高斯球位置张量 [N, 3]
            feats: 特征张量 [N, feat_dim]
        """
        key = (region, moment)
        if key not in self._storage:
            self._storage[key] = {}
        
        position_keys = self._tensor_to_keys(anchor_positions)
        for i, pos_key in enumerate(position_keys):
            self._storage[key][pos_key] = feats[i]
        
        # 清除缓存
        if key in self._tensor_cache:
            del self._tensor_cache[key]
    
    def merge_multi_region_features(self, region_data_dict: dict):
        """
        合并多个区域的特征
        Args:
            region_data_dict: {region_id: {
                'anchor_feat_dict': {(r, m): feats_tensor},  # 该区域内的特征字典
                'anchor_positions': Tensor [N, 3],           # 该区域内高斯的位置
            }}
        """
        for region_id, region_data in region_data_dict.items():
            feat_dict = region_data.get('anchor_feat_dict', {})
            anchor_positions = region_data.get('anchor_positions', None)
            
            for (r, m), feats in feat_dict.items():
                if anchor_positions is not None:
                    self.add_region_features_batch(r, m, anchor_positions, feats)
                else:
                    # 如果没有提供位置，从特征数量推断（不推荐）
                    print(f"Warning: No anchor positions provided for region {region_id}")
        
        print(f"RenderingAnchorFeatStorage: Merged features from {len(region_data_dict)} regions")
    
    def get(self, region: int, moment: int, anchor_position=None):
        """
        获取特征
        Args:
            region: 区域标识
            moment: 时刻标识
            anchor_position: 高斯球位置，如果为None则返回该(region, moment)下的所有特征
        Returns:
            如果 anchor_position 为None: {position_key: feat} 字典
            否则: 单个特征张量 [feat_dim]
        """
        key = (region, moment)
        if key not in self._storage:
            raise KeyError(f"Features for (region={region}, moment={moment}) not found")
        
        if anchor_position is None:
            return self._storage[key]
        else:
            position_key = self._position_to_key(anchor_position)
            if position_key not in self._storage[key]:
                raise KeyError(f"Feature for (region={region}, moment={moment}, pos={position_key}) not found")
            return self._storage[key][position_key]
    
    def get_batch(self, region: int, moment: int, anchor_positions):
        """
        批量获取特征 - 核心方法
        Args:
            region: 区域标识
            moment: 时刻标识
            anchor_positions: 高斯球位置张量 [N, 3]
        Returns:
            特征张量 [N, feat_dim]
        """
        key = (region, moment)
        if key not in self._storage:
            raise KeyError(f"Features for (region={region}, moment={moment}) not found")
        
        # 构建或更新缓存
        if key not in self._tensor_cache:
            poses_list = list(self._storage[key].keys())
            feats_list = [self._storage[key][pos] for pos in poses_list]
            self._tensor_cache[key] = {
                'poses': poses_list,
                'feats': torch.stack(feats_list, dim=0)
            }
        
        # 将查询位置转换为键
        query_keys = self._tensor_to_keys(anchor_positions)
        
        # 创建位置到索引的映射
        cache_poses = self._tensor_cache[key]['poses']
        pos_to_idx = {pos: i for i, pos in enumerate(cache_poses)}
        
        # 获取所需的索引，处理找不到的情况
        indices = []
        for q_key in query_keys:
            if q_key in pos_to_idx:
                indices.append(pos_to_idx[q_key])
            else:
                # 找不到时返回None或零向量，这里用零向量
                indices.append(-1)
        
        # 获取特征，处理-1的情况
        result_feats = []
        for idx in indices:
            if idx >= 0:
                result_feats.append(self._tensor_cache[key]['feats'][idx])
            else:
                # 找不到的位置返回零向量
                result_feats.append(torch.zeros(self.feat_dim, device=self.device))
        
        return torch.stack(result_feats, dim=0)
    
    def get_by_triple_key(self, region: int, moment: int, anchor_position):
        """
        通过完整的三元键 (区域, 时间, 高斯位置) 获取特征
        """
        return self.get(region, moment, anchor_position)
    
    def keys(self):
        """获取所有 (region, moment) 键"""
        return self._storage.keys()
    
    def get_all_triple_keys(self):
        """获取所有完整的三元键 (region, moment, anchor_position)"""
        triple_keys = []
        for (r, m), feat_dict in self._storage.items():
            for pos in feat_dict.keys():
                triple_keys.append((r, m, pos))
        return triple_keys
    
    def __len__(self):
        """返回所有特征的总数"""
        total = 0
        for feat_dict in self._storage.values():
            total += len(feat_dict)
        return total
    
    def __contains__(self, key):
        """
        检查键是否存在
        key可以是：
            - (region, moment) 二元组
            - (region, moment, anchor_position) 三元组
        """
        if len(key) == 2:
            return key in self._storage
        elif len(key) == 3:
            r, m, pos = key
            if (r, m) not in self._storage:
                return False
            pos_key = self._position_to_key(pos) if not isinstance(pos, tuple) else pos
            return pos_key in self._storage[(r, m)]
        return False
    
    def save(self, path: str):
        """
        保存特征存储
        """
        save_dir = os.path.dirname(path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        data = {}
        total_features = 0
        
        for (region, moment), feat_dict in self._storage.items():
            anchor_poses = list(feat_dict.keys())
            feats_list = [feat_dict[pos] for pos in anchor_poses]
            feats_tensor = torch.stack(feats_list, dim=0) if feats_list else torch.empty(0, self.feat_dim)
            
            data[(region, moment)] = {
                'anchor_poses': anchor_poses,
                'feats': feats_tensor.detach().cpu()
            }
            total_features += len(anchor_poses)
        
        save_data = {
            'version': 2,  # 新版本号，标识使用位置作为键
            'feat_dim': self.feat_dim,
            'device': self.device,
            'precision': self.precision,
            'total_features': total_features,
            'data': data
        }
        
        torch.save(save_data, path)
        print(f"RenderingAnchorFeatStorage: Saved {total_features} features to {path}")
        return save_data
    
    def load(self, path: str, device='cuda', strict=True):
        """
        加载特征存储
        """
        if not os.path.exists(path):
            if strict:
                raise FileNotFoundError(f"RenderingAnchorFeatStorage: File not found at {path}")
            else:
                print(f"RenderingAnchorFeatStorage: File not found at {path}")
                return
        
        load_data = torch.load(path, map_location='cpu')
        
        version = load_data.get('version', 0)
        if version < 2 and strict:
            print(f"Warning: Loading older format (version {version}), position-based keys may not be compatible")
        
        self.feat_dim = load_data.get('feat_dim', self.feat_dim)
        self.device = device
        self.precision = load_data.get('precision', 6)
        
        self._storage = {}
        total_loaded = 0
        data = load_data.get('data', {})
        
        for (region, moment), entry in data.items():
            anchor_poses = entry.get('anchor_poses', [])
            feats_tensor = entry.get('feats', torch.empty(0, self.feat_dim))
            
            feats_tensor = feats_tensor.to(device)
            
            self._storage[(region, moment)] = {}
            for i, pos in enumerate(anchor_poses):
                self._storage[(region, moment)][pos] = feats_tensor[i]
            
            total_loaded += len(anchor_poses)
        
        self._tensor_cache = {}
        
        print(f"RenderingAnchorFeatStorage: Loaded {total_loaded} features from {path}")
        return load_data
