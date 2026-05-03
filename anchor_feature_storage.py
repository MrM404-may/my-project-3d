import os
import torch
import json
import numpy as np


class AnchorFeatureStorage:
    """
    用于存储高斯球的_anchor_feat属性，使用字典结构，键为(region, moment, position)
    支持增量式保存和加载
    """
    def __init__(self, save_path, format='pt'):
        """
        初始化存储
        参数:
            save_path: 保存路径
            format: 保存格式，'pt'或'json'，默认为'pt'
        """
        self.save_path = save_path
        self.format = format.lower()
        self.storage_dict = {}  # 存储字典，键为(region, moment, position_tuple)，值为anchor_feat
        self.positions_dict = {}  # 存储位置数据，键为(region, moment)，值为numpy数组
        self.features_dict = {}  # 存储特征数据，键为(region, moment)，值为numpy数组
        self.keys_dict = {}  # 存储原始键，键为(region, moment)，值为键列表
        
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # 尝试加载已有的数据
        self._load_existing()
    
    def _build_cache(self):
        """构建位置和特征的缓存，以便快速查找"""
        self.positions_dict = {}
        self.features_dict = {}
        self.keys_dict = {}
        
        for key, feat in self.storage_dict.items():
            region, moment, pos = key
            cache_key = (region, moment)
            
            if cache_key not in self.positions_dict:
                self.positions_dict[cache_key] = []
                self.features_dict[cache_key] = []
                self.keys_dict[cache_key] = []
            
            self.positions_dict[cache_key].append(pos)
            self.features_dict[cache_key].append(feat)
            self.keys_dict[cache_key].append(key)
        
        # 转换为 numpy 数组
        for cache_key in self.positions_dict.keys():
            self.positions_dict[cache_key] = np.array(self.positions_dict[cache_key])
    
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
        
        # 重建缓存
        self._build_cache()
    
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
        if cache_key not in self.positions_dict:
            # 没有该 region 和 moment 的数据
            return None, torch.zeros(query_positions.shape[0], dtype=torch.bool, device=query_positions.device if isinstance(query_positions, torch.Tensor) else 'cpu')
        
        stored_positions = self.positions_dict[cache_key]
        stored_features = self.features_dict[cache_key]
        
        # 将 query_positions 转换为 numpy
        if isinstance(query_positions, torch.Tensor):
            query_np = query_positions.detach().cpu().numpy()
            device = query_positions.device
        else:
            query_np = np.array(query_positions)
            device = 'cpu'
        
        # 计算所有查询点与存储点的距离
        # 使用广播计算
        # stored_positions: [M, 3], query_np: [N, 3]
        # 距离矩阵: [N, M]
        diff = query_np[:, np.newaxis, :] - stored_positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        # 找到每个查询点的最小距离和对应的索引
        min_distances = np.min(distances, axis=1)
        closest_indices = np.argmin(distances, axis=1)
        
        # 确定哪些匹配在容差范围内
        match_mask = min_distances <= tolerance
        
        # 准备输出特征
        feat_dim = len(stored_features[0]) if stored_features else 0
        assigned_feats_np = np.zeros((query_positions.shape[0], feat_dim), dtype=np.float32)
        
        for i in range(query_positions.shape[0]):
            if match_mask[i]:
                if self.format == 'json':
                    assigned_feats_np[i] = np.array(stored_features[closest_indices[i]])
                else:
                    assigned_feats_np[i] = stored_features[closest_indices[i]].numpy()
        
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
                self.storage_dict = torch.load(self.save_path)
                print(f"[AnchorFeatureStorage] Loaded {len(self.storage_dict)} entries from {self.save_path}")
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
            
            # 构建缓存
            self._build_cache()
        except Exception as e:
            print(f"[AnchorFeatureStorage] Error loading existing file: {e}, starting fresh")
            self.storage_dict = {}
    
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
        self.keys_dict = {}
    
    def __len__(self):
        return len(self.storage_dict)
