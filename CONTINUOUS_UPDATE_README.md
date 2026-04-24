# 持续更新式建模 (Continuous Update Modeling)

本代码库基于GaussianUpdate论文的思想和现有的分区MLP系统，实现了持续更新式建模功能。

## 概述

持续更新系统允许在新时间戳的数据可用时，逐步更新高斯模型，同时保留之前学习到的信息。系统的核心是一个三阶段更新流程，结合了全局外观模型、布局不变掩码、移除因子等技术。

## 新增文件

### 1. `scene/continuous_update.py`
核心持续更新模块，包含以下主要组件：

- **HashEncoding**: 5D哈希编码（位置+时间+区域）
- **GlobalAppearanceModel**: 全局外观模型，结合哈希编码和小MLP
- **VisibilityPool**: 可见性池，用于存储历史状态和相机信息
- **LayoutInvariantMaskGenerator**: 布局不变掩码生成器（SAM接口占位）
- **RemovalFactor**: 移除因子，用于标记消失的高斯
- **DBSCANPruner**: DBSCAN聚类剪枝
- **ContinuousUpdateManager**: 主管理器，协调三阶段更新

### 2. `train_continuous_update.py`
持续更新训练脚本，展示了如何使用新功能进行多时间戳训练。

## 修改的文件

### 1. `scene/gaussian_model.py`
- 新增持续更新相关成员变量
- 新增持续更新方法：
  - `init_continuous_update()`: 初始化管理器
  - `set_timestamp()`: 设置当前时间戳
  - `init_timestamp_update()`: 初始化新时间戳
  - `get_appearance_updates()`: 获取外观更新
  - `apply_removal_factor()`: 应用移除因子
  - `get_removal_regularization_loss()`: 获取正则化损失
  - `update_training_stage()`: 更新训练阶段
  - `save_current_state_to_visibility_pool()`: 保存状态
  - `generate_layout_invariant_mask()`: 生成布局不变掩码
  - `save_continuous_update_checkpoint()` / `load_continuous_update_checkpoint()`: 检查点保存/加载

### 2. `gaussian_renderer/__init__.py`
- 为`generate_neural_gaussians`和`render`函数添加`use_continuous_update`参数
- 新增应用全局外观模型、移除因子、外观更新的代码

## 三阶段更新流程

### 1. 外观更新阶段（Stage 1: Global Appearance Update）
- **持续时间**: 7000次迭代
- **目标**: 冻结几何，仅更新全局外观模型
- **特点**: 可以使用布局不变掩码，仅训练变化区域

### 2. 几何更新阶段（Stage 2: Geometry and Layout Update）
- **持续时间**: 8000次迭代
- **目标**: 允许几何更新，学习移除因子
- **DBSCAN剪枝**: 在迭代5000和8000时执行DBSCAN聚类剪枝

### 3. 联合精炼阶段（Stage 3: Joint Refinement）
- **持续时间**: 15000次迭代
- **目标**: 精炼所有参数
- **重要性剪枝**: 在迭代4000时执行

## 如何使用

### 基本用法

1. **初始化持续更新管理器**：
```python
gaussians.init_continuous_update(num_regions=65)
```

2. **设置时间戳**：
```python
# 时间戳应该归一化到[0, 1]范围
gaussians.set_timestamp(normalized_timestamp)
```

3. **初始化新时间戳**：
```python
gaussians.init_timestamp_update(timestamp)
```

4. **训练过程中更新阶段**：
```python
# 每步迭代后调用
gaussians.update_training_stage()
```

5. **渲染时启用持续更新**：
```python
render_pkg = render(
    viewpoint_cam, gaussians, pipe, background,
    visible_mask=voxel_visible_mask,
    camera_region=camera_region,
    use_continuous_update=True  # 启用持续更新
)
```

6. **使用训练脚本**：
```bash
python train_continuous_update.py \
    -s <source_path> \
    -m <model_path> \
    --num_timestamps 5 \
    --timestamp_duration 30000 \
    --num_regions 65
```

### 关键参数

- `num_timestamps`: 时间戳数量
- `timestamp_duration`: 每个时间戳的训练迭代次数
- `num_regions`: 区域数量

## SAM集成说明

目前`LayoutInvariantMaskGenerator`中的SAM接口是占位实现，使用了简单的图像差异作为替代。后续需要实现真实的SAM网络集成：

1. 下载SAM模型权重
2. 实现`load_sam_model()`方法
3. 在`generate_mask()`中添加真实的实例分割和IoU计算逻辑

## 主要改进点

1. **时间感知**: 哈希编码输入包含时间戳，支持时序建模
2. **区域感知**: 哈希编码输入包含区域ID，与现有分区系统集成
3. **鲁棒更新**: 三阶段流程确保更新的稳定性
4. **消失物体处理**: 移除因子和DBSCAN剪枝
5. **历史信息保留**: 可见性池和生成式回放框架

## 注意事项

1. 代码中的SAM接口需要后续完善
2. 部分实现是简化版本，可根据需要进一步优化
3. DBSCAN的参数（eps, min_samples）可能需要调整以适应具体场景
4. 移除因子的正则化损失权重需要调整

## 相关文件

- `GaussianUpdate_详解.md`: 原始GaussianUpdate论文的详细解析
