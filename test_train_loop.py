"""
测试训练循环中各个操作的开销
"""

import time
import random

def test_dict_lookup_vs_list_access():
    """测试字典查找 vs 列表访问"""
    print("\n=== 测试1: 字典查找 vs 列表访问 ===")
    
    # 模拟有100个相机的场景
    num_cameras = 100
    camera_ids = list(range(num_cameras))
    random.shuffle(camera_ids)
    
    # 模拟相机ID到相机对象的映射（你的版本）
    camera_id_map = {cid: {"id": cid, "center": [random.random() for _ in range(3)]} for cid in camera_ids}
    
    # 模拟相机列表（原版）
    camera_list = [camera_id_map[cid] for cid in camera_ids]
    
    # 测试字典查找
    num_iterations = 10000
    camera_sequence = random.choices(camera_ids, k=num_iterations)
    
    start = time.time()
    for cid in camera_sequence:
        cam = camera_id_map[cid]  # 字典查找
    end = time.time()
    dict_time = end - start
    print(f"字典查找: {dict_time:.4f} 秒 ({dict_time/num_iterations*1000:.4f} ms/次)")
    
    # 测试列表索引
    indices = random.choices(range(num_cameras), k=num_iterations)
    start = time.time()
    for idx in indices:
        cam = camera_list[idx]  # 列表索引
    end = time.time()
    list_time = end - start
    print(f"列表索引: {list_time:.4f} 秒 ({list_time/num_iterations*1000:.4f} ms/次)")
    
    print(f"字典查找相对于列表索引: {dict_time/list_time:.2f}x")


def test_region_switch_overhead():
    """测试区域切换检查的开销"""
    print("\n=== 测试2: 区域切换检查开销 ===")
    
    num_regions = 65
    total_iterations = 30000
    
    # 创建迭代边界
    region_bounds = []
    start = 0
    for i in range(num_regions):
        iters = random.randint(300, 500)
        end = start + iters
        region_bounds.append((start, end))
        start = end
    region_bounds[-1] = (region_bounds[-1][0], total_iterations)
    
    current_region = 0
    
    # 测试：每次迭代都检查所有区域（你的版本）
    num_test = 1000
    start = time.time()
    for iteration in range(num_test):
        for region_idx in range(num_regions):
            start_iter, end_iter = region_bounds[region_idx]
            if iteration == start_iter and region_idx != current_region:
                current_region = region_idx
    end = time.time()
    full_check_time = end - start
    print(f"每次迭代检查所有区域: {full_check_time:.4f} 秒 (共 {num_test} 次)")
    print(f"平均每次迭代: {full_check_time/num_test*1000:.4f} ms")
    
    # 测试：只检查当前区域的边界（优化方案）
    current_start, current_end = region_bounds[0]
    start = time.time()
    for iteration in range(num_test):
        if iteration == current_end:
            current_region += 1
            current_start, current_end = region_bounds[current_region]
    end = time.time()
    optimized_time = end - start
    print(f"只检查当前区域边界: {optimized_time:.4f} 秒 (共 {num_test} 次)")
    print(f"平均每次迭代: {optimized_time/num_test*1000:.4f} ms")
    
    print(f"优化效果: {full_check_time/optimized_time:.2f}x")


def test_tensor_concatenation():
    """测试张量拼接的开销"""
    print("\n=== 测试3: 张量拼接开销 ===")
    
    # 模拟统计量扩展（你的版本中的 adjust_anchor）
    num_anchors = 500000
    num_new_anchors = 1000
    n_offsets = 8
    
    opacity_accum = [0.0] * num_anchors
    offset_gradient_accum = [0.0] * (num_anchors * n_offsets)
    offset_denom = [0.0] * (num_anchors * n_offsets)
    
    # 测试 torch.cat 风格的扩展
    num_iterations = 100
    start = time.time()
    for _ in range(num_iterations):
        # 模拟 torch.cat
        padding_opacity = [0.0] * num_new_anchors
        new_opacity_accum = opacity_accum + padding_opacity
        
        padding_gradient = [0.0] * (num_new_anchors * n_offsets)
        new_offset_gradient_accum = offset_gradient_accum + padding_gradient
        new_offset_denom = offset_denom + padding_gradient
    end = time.time()
    
    print(f"统计量扩展 ({num_iterations} 次): {end-start:.4f} 秒")
    print(f"平均每次扩展: {(end-start)/num_iterations*1000:.4f} ms")


if __name__ == "__main__":
    test_dict_lookup_vs_list_access()
    test_region_switch_overhead()
    test_tensor_concatenation()
