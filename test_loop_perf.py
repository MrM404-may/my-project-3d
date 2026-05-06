import time

def benchmark_loop(num_elements, num_iterations=1000):
    """
    测试不同元素数量下循环的性能
    
    参数:
        num_elements: 元素数量
        num_iterations: 迭代次数
    """
    print(f"\n=== 测试 {num_elements} 个元素 ===")
    
    # 创建模拟参数列表
    params = [{'grad': None} for _ in range(num_elements)]
    
    # 测试: 模拟优化器 step() 的遍历
    print("测试: 遍历所有元素并检查 grad")
    start = time.time()
    for _ in range(num_iterations):
        for p in params:
            if p['grad'] is None:
                continue
            # 如果有梯度，模拟更新
            pass
    end = time.time()
    print(f"  耗时: {end-start:.4f} 秒")
    print(f"  平均每轮: {(end-start)/num_iterations*1000:.4f} 毫秒")
    print(f"  总检查次数: {num_elements * num_iterations:,}")

if __name__ == "__main__":
    # 测试不同元素数量
    benchmark_loop(3)        # 原版（约3个MLP）
    benchmark_loop(195)      # 你的版本（65个区域 × 3个MLP）
    benchmark_loop(1950)     # 更大的测试
    benchmark_loop(482560)   # 实际参数数量（65区域 × 每个区域~7424参数）
