# 4D Hash + Tiny MLP 替代区域 MLP 使用指南

## 概述

本实现用一个全局的 4D Hash + Tiny MLP 替代了原来的区域 MLP。哈希编码输入包括：
- 位置 (x, y, z)
- 时间戳 (t)
- 区域 ID (region)

## 架构

### 1. 全局高斯 MLP (`GlobalGaussianMLP`)
- **输入**：位置 + 时间 + 区域（哈希编码） + 视角 + 外观嵌入
- **输出**：不透明度、协方差参数、颜色

### 2. 持续更新管理器 (`ContinuousUpdateManager`)
- 统一管理更新过程
- 协调移除因子
- 管理时间戳

## 使用方法

### 初始化

在创建 GaussianModel 时，默认启用全局 MLP：
```python
gaussians = GaussianModel(
    num_regions=65,
    use_global_mlp=True  # 默认为 True
)
```

### 设置时间戳
```python
gaussians.set_timestamp(timestamp)  # timestamp 归一化到 [0, 1]
```

### 训练

使用新的训练脚本或修改原脚本：
```bash
python train_continuous_update.py --source_path <path> --num_timestamps 5
```

### 渲染

在渲染时设置 `use_continuous_update=True`：
```python
render_pkg = render(
    viewpoint_camera, 
    gaussians, 
    pipe, 
    bg_color,
    camera_region=0,
    use_continuous_update=True
)
```

## 参数

### 哈希编码参数
- `num_levels`：哈希层级数（默认 16）
- `min_res`：最粗分辨率（默认 16）
- `max_res`：最细分辨率（默认 512）
- `log2_hashmap_size`：哈希表大小（默认 21，即 2^21）

### 持续更新参数
- `num_timestamps`：时间戳数量
- `timestamp_duration`：每个时间戳的迭代次数
- `use_continuous_update`：是否启用持续更新

## 关键文件

### 新增文件
- `scene/continuous_update.py`：核心实现
- `train_continuous_update.py`：训练脚本
- `USAGE_GUIDE.md`：本文档
- `CONTINUOUS_UPDATE_README.md`：详细说明

### 修改文件
- `scene/gaussian_model.py`：集成全局 MLP
- `gaussian_renderer/__init__.py`：修改渲染流程

## 向后兼容

- 保留了原有的区域 MLP 架构
- 可以通过设置 `use_global_mlp=False` 回退到原有架构
- 所有现有接口保持不变

## TODO

1. 完善 SAM 网络集成
2. 优化哈希编码性能
3. 添加更多的可视化和调试工具
