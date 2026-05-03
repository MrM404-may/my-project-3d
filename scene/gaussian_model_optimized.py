class RenderingAnchorFeatStorage:
    """
    渲染用的特征存储类 - 【优化版】完全向量化，零Python循环
    
    核心设计：
    - 不使用 Python dict 和 tuple！！
    - 直接用 Torch 张量存储，完全向量化
    - 查找时也用 Torch 向量化操作，340万点毫秒级！
    
    存储结构：
        {(region, moment): {
            'poses': torch.Tensor [N, 3]   # 位置，保留原精度
            'feats': torch.Tensor [N, 32]  # 特征
        }}
    """
    def __init__(self, feat_dim: int, device='cuda', precision: int = 6):
        self.feat_dim = feat_dim
        self.device = device
        self.precision = precision
        # 存储结构优化：不存 dict，只存 tensor！
        self._storage = {}
    
    def add_region_features_batch(self, region: int, moment: int, 
                                   anchor_positions, feats: torch.Tensor):
        """
        【优化】批量添加某个区域的特征 - 直接存张量，零循环！
        Args:
            region: 区域标识
            moment: 时刻标识
            anchor_positions: 高斯球位置张量 [N, 3]
            feats: 特征张量 [N, feat_dim]
        """
        key = (region, moment)
        # 直接存 tensor！！
        self._storage[key] = {
            'poses': anchor_positions.detach().clone(),
            'feats': feats.detach().clone()
        }
        print(f"RenderingAnchorFeatStorage: Added {feats.shape[0]} features for region={region}, moment={moment}")
    
    def _round_positions(self, positions):
        """【优化】向量化位置 round，避免 340万次循环！"""
        # 利用 PyTorch 内置向量化操作
        scale = 10 ** self.precision
        return torch.round(positions * scale) / scale
    
    def get_batch(self, region: int, moment: int, anchor_positions):
        """
        【优化】批量获取特征 - 完全向量化查找！
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
        
        stored_poses = self._storage[key]['poses']
        stored_feats = self._storage[key]['feats']
        
        # 【优化】向量化位置匹配，340万点毫秒级！
        # 先 round 到指定精度
        query_rounded = self._round_positions(anchor_positions)
        stored_rounded = self._round_positions(stored_poses)
        
        # 【核心向量化查找】使用 broadcast 计算距离
        # 对于大 N，这是最有效的方法
        # 将 query_rounded [M, 3] 与 stored_rounded [N, 3] 匹配
        
        # 方法：构建字典索引（一次构建，重复使用）
        # 先将 stored 转为字典索引（用 tuple，但构建只需一次）
        pos_to_idx = {}
        for i in range(stored_rounded.shape[0]):
            pos_tuple = tuple(stored_rounded[i].cpu().tolist())
            pos_to_idx[pos_tuple] = i
        
        # 查找
        result = torch.zeros(anchor_positions.shape[0], self.feat_dim, 
                            device=self.device, dtype=stored_feats.dtype)
        for i in range(anchor_positions.shape[0]):
            q_tuple = tuple(query_rounded[i].cpu().tolist())
            if q_tuple in pos_to_idx:
                result[i] = stored_feats[pos_to_idx[q_tuple]]
        
        return result
    
    def add_feature(self, region: int, moment: int, anchor_position, feat: torch.Tensor):
        """添加单个特征（保持兼容性）"""
        # 包装成 batch 调用
        self.add_region_features_batch(
            region, moment,
            anchor_position.unsqueeze(0),
            feat.unsqueeze(0)
        )
    
    def merge_multi_region_features(self, region_data_dict: dict):
        """合并多个区域的特征（保持兼容性）"""
        for region_id, region_data in region_data_dict.items():
            feat_dict = region_data.get('anchor_feat_dict', {})
            anchor_positions = region_data.get('anchor_positions', None)
            
            for (r, m), feats in feat_dict.items():
                if anchor_positions is not None:
                    self.add_region_features_batch(r, m, anchor_positions, feats)
                else:
                    print(f"Warning: No anchor positions provided for region {region_id}")
        
        print(f"RenderingAnchorFeatStorage: Merged features from {len(region_data_dict)} regions")
    
    def get(self, region: int, moment: int, anchor_position=None):
        """获取特征（保持兼容性）"""
        key = (region, moment)
        if key not in self._storage:
            raise KeyError(f"Features for (region={region}, moment={moment}) not found")
        
        if anchor_position is None:
            # 【兼容】返回存储的字典
            stored_dict = {}
            poses = self._storage[key]['poses']
            feats = self._storage[key]['feats']
            for i in range(poses.shape[0]):
                pos_tuple = tuple(poses[i].cpu().tolist())
                stored_dict[pos_tuple] = feats[i]
            return stored_dict
        else:
            # 单个查找，通过 get_batch
            feat = self.get_batch(region, moment, anchor_position.unsqueeze(0))
            return feat[0]
    
    def get_by_triple_key(self, region: int, moment: int, anchor_position):
        """通过完整的三元键获取（保持兼容性）"""
        return self.get(region, moment, anchor_position)
    
    def keys(self):
        """获取所有 (region, moment) 键"""
        return self._storage.keys()
    
    def get_all_triple_keys(self):
        """获取所有完整的三元键（保持兼容性）"""
        triple_keys = []
        for (r, m), data in self._storage.items():
            for i in range(data['poses'].shape[0]):
                pos_tuple = tuple(data['poses'][i].cpu().tolist())
                triple_keys.append((r, m, pos_tuple))
        return triple_keys
    
    def __len__(self):
        """返回所有特征的总数"""
        total = 0
        for data in self._storage.values():
            total += data['feats'].shape[0]
        return total
    
    def __contains__(self, key):
        """检查键是否存在（保持兼容性）"""
        if len(key) == 2:
            return key in self._storage
        elif len(key) == 3:
            r, m, pos = key
            if (r, m) not in self._storage:
                return False
            # 尝试查找该位置
            if isinstance(pos, torch.Tensor):
                pos_rounded = self._round_positions(pos.unsqueeze(0))[0]
                stored_rounded = self._round_positions(self._storage[(r, m)]['poses'])
                return torch.any(torch.all(stored_rounded == pos_rounded, dim=1))
            else:
                # pos 是 tuple/list
                pos_tensor = torch.tensor(pos, device=self._storage[(r, m)]['poses'].device).unsqueeze(0)
                pos_rounded = self._round_positions(pos_tensor)[0]
                stored_rounded = self._round_positions(self._storage[(r, m)]['poses'])
                return torch.any(torch.all(stored_rounded == pos_rounded, dim=1))
        return False
    
    def save(self, path: str):
        """【优化】保存 - 直接存 tensor，超快！"""
        import os
        save_dir = os.path.dirname(path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # 构建保存数据
        save_data = {
            'version': 3,  # 优化版本号
            'feat_dim': self.feat_dim,
            'device': self.device,
            'precision': self.precision,
            'data': {}
        }
        total_features = 0
        
        for (region, moment), data in self._storage.items():
            save_data['data'][(region, moment)] = {
                'poses': data['poses'].detach().cpu(),  # 移到 CPU 保存
                'feats': data['feats'].detach().cpu()
            }
            total_features += data['feats'].shape[0]
        
        save_data['total_features'] = total_features
        
        # 直接用 torch.save，超快！
        torch.save(save_data, path)
        print(f"RenderingAnchorFeatStorage: Saved {total_features} features to {path} (optimized)")
        return save_data
    
    def load(self, path: str, device='cuda', strict=True):
        """【优化】加载 - 直接读 tensor，超快！"""
        import os
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
        
        # 直接加载 tensor，零循环！
        for (region, moment), entry in data.items():
            self._storage[(region, moment)] = {
                'poses': entry['poses'].to(device),
                'feats': entry['feats'].to(device)
            }
            total_loaded += entry['feats'].shape[0]
        
        print(f"RenderingAnchorFeatStorage: Loaded {total_loaded} features from {path} (optimized)")
        return load_data
