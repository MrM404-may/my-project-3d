# Anchor Feature 存储和渲染功能使用说明

## 概述

本功能实现了在分区训练过程中保存高斯球的 `_anchor_feat` 属性，并在渲染时根据 `region` 和 `moment` 参数加载和应用对应的特征。支持增量式保存和加载。

## 文件结构

1. `anchor_feature_storage.py` - 核心存储类
2. `scene/gaussian_model.py` - 高斯模型类（已添加相关方法）
3. `train9pro-Ma0421.py` - 训练脚本（已集成存储功能）
4. `render9-huang.py` - 渲染脚本（已集成加载和应用功能）
5. `test_anchor_feature_storage.py` - 测试脚本

## 功能特性

### 训练时存储
- 以字典形式存储，键为 `(region, moment, position)`
- 值为对应的 `_anchor_feat`
- 支持 `.pt` (PyTorch) 和 `.json` 两种保存格式
- 增量式保存：每次区域训练完成后在原有基础上添加新数据
- 自动加载已有数据

### 渲染时加载和应用
- 支持通过命令行参数指定 `region`, `moment`, 和 anchor feature 文件
- 根据高斯球位置自动匹配并应用对应的 `_anchor_feat`
- 区域切换时自动更新特征
- 支持位置容差匹配

## 使用方法

### 1. 训练时自动保存

在 `train9pro-Ma0421.py` 中已经集成了自动保存功能：

- 在训练开始时会自动初始化存储
- 在区域切换时会自动保存当前区域的 anchor features
- 在训练结束时会自动保存最后一个区域的 anchor features

保存路径默认设置为：`{dataset.model_path}/anchor_features.pt`

### 2. 修改训练时的保存格式

如果需要修改保存格式，可以在 `train9pro-Ma0421.py` 中修改：

```python
# 修改前
anchor_feat_save_path = os.path.join(dataset.model_path, "anchor_features.pt")
gaussians.init_anchor_feature_storage(anchor_feat_save_path, format='pt')

# 修改为 json 格式
anchor_feat_save_path = os.path.join(dataset.model_path, "anchor_features.json")
gaussians.init_anchor_feature_storage(anchor_feat_save_path, format='json')
```

### 3. 修改训练时的 moment 值

如果需要为不同的训练阶段使用不同的 moment 值，可以在保存时修改：

```python
# 在区域切换时
gaussians.save_current_anchor_features(prev_region_idx, moment=1)  # 修改 moment 值

# 在训练结束时
gaussians.save_current_anchor_features(current_region_idx, moment=1)
```

### 4. 渲染时加载和应用 anchor features

使用 `render9-huang.py` 进行渲染时，可以通过以下命令行参数：

```bash
# 基本用法
python render9-huang.py --model_path /path/to/model --anchor_feat_file /path/to/anchor_features.pt

# 指定 region 和 moment
python render9-huang.py --model_path /path/to/model --anchor_feat_file /path/to/anchor_features.pt --region 0 --moment 0

# 完整参数示例
python render9-huang.py \
    --model_path /path/to/model \
    --iteration 30000 \
    --anchor_feat_file /path/to/anchor_features.pt \
    --region 0 \
    --moment 0 \
    --skip_train
```

**渲染参数说明：**
- `--anchor_feat_file`: anchor features 文件路径（.pt 或 .json）
- `--region`: 覆盖渲染时的区域（可选，不指定则使用相机对应区域）
- `--moment`: 使用的时刻值（默认为 0）

### 5. 独立使用 AnchorFeatureStorage 类

如果需要在其他地方使用，可以参考以下代码：

```python
from anchor_feature_storage import AnchorFeatureStorage

# 初始化
storage = AnchorFeatureStorage("path/to/save.pt", format='pt')

# 添加数据
storage.add(
    region=0,
    moment=0,
    positions=positions_tensor,  # 形状为 [N, 3]
    anchor_feats=feats_tensor    # 形状为 [N, feat_dim]
)

# 保存到文件
storage.save()

# 加载已有数据
new_storage = AnchorFeatureStorage("path/to/save.pt", format='pt')

# 根据位置匹配和获取特征（用于渲染）
assigned_feats, match_mask = new_storage.match_and_assign_features(
    region=0,
    moment=0,
    query_positions=query_positions_tensor,  # 需要匹配的位置
    tolerance=1e-3  # 匹配容差
)

# 获取特定区域和时刻的所有数据
positions_list, feats_list = new_storage.get_region_moment(region=0, moment=0)

# 获取单个高斯球的特征
feat = new_storage.get(region=0, moment=0, position=position_tensor)
```

## 数据结构

### 存储键格式

存储字典的键为 `(region_id, moment_id, (x, y, z))`：
- `region_id` - 区域索引（整数）
- `moment_id` - 时刻值（整数，如 0, 1, 2, 3）
- `(x, y, z)` - 高斯球的 3D 坐标位置

### .pt 文件格式

直接保存 Python 字典：
```python
{
    (0, 0, (1.0, 2.0, 3.0)): torch.Tensor([...]),
    (0, 0, (4.0, 5.0, 6.0)): torch.Tensor([...]),
    ...
}
```

### .json 文件格式

键转换为字符串，特征转换为列表：
```json
{
    "0,0,1.0,2.0,3.0": [...],
    "0,0,4.0,5.0,6.0": [...],
    ...
}
```

## API 说明

### AnchorFeatureStorage 类

| 方法 | 说明 |
|------|------|
| `__init__(save_path, format='pt')` | 初始化存储 |
| `add(region, moment, positions, anchor_feats)` | 添加数据 |
| `get(region, moment, position)` | 获取单个特征 |
| `get_region_moment(region, moment)` | 获取区域和时刻的所有数据 |
| `match_and_assign_features(region, moment, query_positions, tolerance=1e-3)` | 根据位置匹配并返回特征 |
| `save()` | 保存到文件 |
| `get_all_regions()` | 获取所有区域 ID |
| `get_all_moments()` | 获取所有时刻值 |
| `clear()` | 清空存储 |

### GaussianModel 类新增方法

| 方法 | 说明 |
|------|------|
| `init_anchor_feature_storage(save_path, format='pt')` | 初始化存储 |
| `save_current_anchor_features(region_id, moment=0)` | 保存当前高斯球的 features |
| `load_anchor_features_from_file(file_path, format='pt')` | 从文件加载 anchor feature 存储 |
| `apply_anchor_features(region, moment, tolerance=1e-3)` | 根据 region 和 moment 匹配并应用 anchor features |

## 注意事项

1. 高斯球位置使用的是考虑了八叉树层级偏移的体素中心位置
2. 增量式保存会自动加载已有文件并添加新数据，不会覆盖旧数据
3. 如果同一键（region, moment, position）的数据重复添加，后添加的会覆盖先添加的
4. 渲染时的位置匹配使用欧几里得距离，默认容差为 1e-3
5. 渲染时只有匹配成功的高斯球会更新 `_anchor_feat`，其他保持不变
