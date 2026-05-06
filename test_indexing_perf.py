"""
测试单索引 vs 双重索引的性能差异

对比:
1. anchor = pc.get_anchor[visible_mask]           (原版 - 单索引)
2. anchor = pc.get_anchor[visible_mask][region_mask] (用户版本 - 双重索引)
"""

import time
import random

def benchmark_indexing(num_elements, num_iterations=100):
    """
    测试索引操作的性能
    
    参数:
        num_elements: 元素数量
        num_iterations: 迭代次数
    """
    print(f"\n=== 测试 {num_elements:,} 个元素 ===")
    
    # 创建模拟数据（使用列表模拟张量）
    anchor = [[random.random() for _ in range(3)] for _ in range(num_elements)]
    feat = [[random.random() for _ in range(32)] for _ in range(num_elements)]
    level = [random.randint(0, 9) for _ in range(num_elements)]
    
    # 创建随机掩码
    visible_mask = [random.random() < 0.7 for _ in range(num_elements)]
    region = [random.randint(0, 64) for _ in range(num_elements)]  # 65个区域
    camera_region = 0
    
    # 预先计算 region_mask（基于 visible_mask 后的结果）
    visible_elements = [i for i, v in enumerate(visible_mask) if v]
    region_mask = [region[i] == camera_region for i in visible_elements]
    
    # 测试1: 单索引（原版）
    print("\n测试1: 单索引 (单重过滤)")
    start = time.time()
    for _ in range(num_iterations):
        anchor_result = [anchor[i] for i, v in enumerate(visible_mask) if v]
        feat_result = [feat[i] for i, v in enumerate(visible_mask) if v]
        level_result = [level[i] for i, v in enumerate(visible_mask) if v]
    end = time.time()
    print(f"  耗时: {end-start:.4f} 秒")
    print(f"  平均每轮: {(end-start)/num_iterations*1000:.4f} 毫秒")
    print(f"  结果大小: {len(anchor_result)} 个元素")
    
    # 测试2: 双重索引（用户版本）
    print("\n测试2: 双重索引 (先visible再region)")
    start = time.time()
    for _ in range(num_iterations):
        # 先应用 visible_mask
        visible_anchor = [anchor[i] for i, v in enumerate(visible_mask) if v]
        visible_feat = [feat[i] for i, v in enumerate(visible_mask) if v]
        visible_level = [level[i] for i, v in enumerate(visible_mask) if v]
        # 再应用 region_mask
        anchor_result = [visible_anchor[i] for i, v in enumerate(region_mask) if v]
        feat_result = [visible_feat[i] for i, v in enumerate(region_mask) if v]
        level_result = [visible_level[i] for i, v in enumerate(region_mask) if v]
    end = time.time()
    print(f"  耗时: {end-start:.4f} 秒")
    print(f"  平均每轮: {(end-start)/num_iterations*1000:.4f} 毫秒")
    print(f"  结果大小: {len(anchor_result)} 个元素")
    
    # 测试3: 预计算组合掩码（优化方案）
    print("\n测试3: 预计算组合掩码 (一次过滤)")
    # 预计算组合掩码
    combined_mask = [False] * num_elements
    visible_idx = 0
    for i in range(num_elements):
        if visible_mask[i]:
            if region_mask[visible_idx]:
                combined_mask[i] = True
            visible_idx += 1
    
    start = time.time()
    for _ in range(num_iterations):
        anchor_result = [anchor[i] for i, v in enumerate(combined_mask) if v]
        feat_result = [feat[i] for i, v in enumerate(combined_mask) if v]
        level_result = [level[i] for i, v in enumerate(combined_mask) if v]
    end = time.time()
    print(f"  耗时: {end-start:.4f} 秒")
    print(f"  平均每轮: {(end-start)/num_iterations*1000:.4f} 毫秒")
    print(f"  结果大小: {len(anchor_result)} 个元素")

if __name__ == "__main__":
    # 测试不同规模的数据
    benchmark_indexing(10000)       # 1万锚点
    benchmark_indexing(50000)       # 5万锚点
    benchmark_indexing(100000)      # 10万锚点
