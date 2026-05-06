"""
测试训练循环中每迭代都执行的关键操作
"""

import time
import random

def test_set_anchor_mask():
    """测试 set_anchor_mask 的开销"""
    print("\n=== 测试 set_anchor_mask 开销 ===")
    
    num_anchors = 500000
    
    # 模拟数据
    anchor = [[random.random() for _ in range(3)] for _ in range(num_anchors)]
    level = [random.randint(0, 9) for _ in range(num_anchors)]
    extra_level = [0.0] * num_anchors
    voxel_size = 0.5
    fork = 2
    standard_dist = 10.0
    resolution_scale = 1.0
    cam_center = [random.random() for _ in range(3)]
    
    num_iterations = 100
    
    start = time.time()
    for _ in range(num_iterations):
        # 模拟 set_anchor_mask 中的计算
        # anchor_pos = anchor + (voxel_size/2) / (fork ** level)
        anchor_pos = [[anchor[i][j] + (voxel_size/2) / (fork ** level[i]) for j in range(3)] for i in range(num_anchors)]
        
        # dist = sqrt(sum((anchor_pos - cam_center)**2)) * resolution_scale
        dist = [(((anchor_pos[i][0]-cam_center[0])**2 + (anchor_pos[i][1]-cam_center[1])**2 + (anchor_pos[i][2]-cam_center[2])**2)**0.5) * resolution_scale for i in range(num_anchors)]
        
        # pred_level = log2(standard_dist/dist) / log2(fork) + extra_level
        pred_level = [(standard_dist/dist[i]) / (fork**0.5) + extra_level[i] for i in range(num_anchors)]
        
        # int_level = round(pred_level)
        int_level = [round(pred_level[i]) for i in range(num_anchors)]
        
        # mask = level <= int_level
        mask = [level[i] <= int_level[i] for i in range(num_anchors)]
        
    end = time.time()
    print(f"set_anchor_mask ({num_iterations} 次): {end-start:.4f} 秒")
    print(f"平均每次: {(end-start)/num_iterations*1000:.2f} ms")


def test_region_check_overhead():
    """测试区域检查的开销"""
    print("\n=== 测试区域检查开销 ===")
    
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
    current_start, current_end = region_bounds[0]
    
    # 测试1：每次迭代都检查所有区域（你的版本）
    print("\n测试1: 每次迭代检查所有区域")
    num_test = 10000
    start = time.time()
    for iteration in range(num_test):
        for region_idx in range(num_regions):
            start_iter, end_iter = region_bounds[region_idx]
            if iteration == start_iter and region_idx != current_region:
                pass  # 模拟区域切换
    end = time.time()
    full_check_time = (end - start) / num_test * 1000
    print(f"平均每次迭代: {full_check_time:.4f} ms")
    
    # 测试2：只检查当前区域边界（优化方案）
    print("\n测试2: 只检查当前区域边界")
    current_region = 0
    current_start, current_end = region_bounds[0]
    
    start = time.time()
    for iteration in range(num_test):
        if iteration == current_end:
            current_region += 1
            current_start, current_end = region_bounds[current_region]
    end = time.time()
    optimized_time = (end - start) / num_test * 1000
    print(f"平均每次迭代: {optimized_time:.4f} ms")
    
    print(f"\n优化效果: {full_check_time/optimized_time:.2f}x (全检查 {full_check_time:.4f}ms vs 优化 {optimized_time:.4f}ms)")


def test_camera_sequence_access():
    """测试相机序列访问的开销"""
    print("\n=== 测试相机序列访问开销 ===")
    
    # 模拟你的版本的相机管理
    num_cameras = 100
    camera_ids = list(range(num_cameras))
    random.shuffle(camera_ids)
    
    # 模拟相机ID到相机对象的映射
    camera_map = {cid: {"id": cid, "center": [random.random() for _ in range(3)]} for cid in camera_ids}
    
    # 模拟相机序列
    camera_sequence = random.choices(camera_ids, k=100000)
    sequence_index = 0
    
    # 测试1：字典查找
    print("\n测试1: 字典查找")
    start = time.time()
    for _ in range(100000):
        camera_id = camera_sequence[sequence_index]
        cam = camera_map[camera_id]
        sequence_index = (sequence_index + 1) % len(camera_sequence)
    end = time.time()
    dict_time = (end - start) / 100000 * 1000
    print(f"平均每次访问: {dict_time:.4f} ms")
    
    # 测试2：列表索引（如果相机对象直接存储在序列中）
    camera_sequence_objects = [camera_map[cid] for cid in camera_ids] * 1000
    sequence_index = 0
    
    print("\n测试2: 直接对象访问")
    start = time.time()
    for _ in range(100000):
        cam = camera_sequence_objects[sequence_index]
        sequence_index = (sequence_index + 1) % len(camera_sequence_objects)
    end = time.time()
    list_time = (end - start) / 100000 * 1000
    print(f"平均每次访问: {list_time:.4f} ms")
    
    print(f"\n字典查找 vs 直接访问: {dict_time/list_time:.2f}x")


def test_local_iter_counter():
    """测试本地迭代计数器的开销"""
    print("\n=== 测试本地迭代计数器开销 ===")
    
    num_regions = 65
    region_local_iters = {i: 0 for i in range(num_regions)}
    region_iters = [random.randint(300, 500) for _ in range(num_regions)]
    
    current_region = 0
    
    # 测试：字典读写
    num_test = 100000
    start = time.time()
    for _ in range(num_test):
        region_local_iters[current_region] += 1
        current_local_iter = region_local_iters[current_region]
    end = time.time()
    
    dict_time = (end - start) / num_test * 1000
    print(f"字典读写平均每次: {dict_time:.4f} ms")
    
    # 测试：简单整数递增
    local_iter = 0
    start = time.time()
    for _ in range(num_test):
        local_iter += 1
    end = time.time()
    
    int_time = (end - start) / num_test * 1000
    print(f"整数递增平均每次: {int_time:.4f} ms")
    
    print(f"\n字典 vs 整数: {dict_time/int_time:.2f}x")


if __name__ == "__main__":
    test_set_anchor_mask()
    test_region_check_overhead()
    test_camera_sequence_access()
    test_local_iter_counter()
