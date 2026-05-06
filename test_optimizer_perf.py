import torch
import time

def benchmark_optimizer(num_params, num_steps=1000):
    """
    测试不同参数数量下优化器的性能
    
    参数:
        num_params: 参数数量
        num_steps: 优化器步骤次数
    """
    print(f"\n=== 测试 {num_params} 个参数 ===")
    
    # 创建参数
    params = [torch.randn(100, requires_grad=True) for _ in range(num_params)]
    optimizer = torch.optim.Adam(params, lr=0.001)
    
    # 测试1: 所有参数都没有梯度
    print("测试1: 所有参数 grad=None")
    start = time.time()
    for _ in range(num_steps):
        optimizer.step()
    end = time.time()
    print(f"  耗时: {end-start:.4f} 秒")
    print(f"  平均每步: {(end-start)/num_steps*1000:.4f} 毫秒")
    
    # 测试2: 前向传播后（有梯度）
    print("\n测试2: 前向传播后（有梯度）")
    start = time.time()
    for _ in range(num_steps):
        # 简单的前向传播
        loss = sum(p.sum() for p in params)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    end = time.time()
    print(f"  耗时: {end-start:.4f} 秒")
    print(f"  平均每步: {(end-start)/num_steps*1000:.4f} 毫秒")

if __name__ == "__main__":
    # 测试不同参数数量
    benchmark_optimizer(3)      # 原版（约3个MLP）
    benchmark_optimizer(195)    # 你的版本（65个区域 × 3个MLP）
    benchmark_optimizer(1950)   # 更大的测试
