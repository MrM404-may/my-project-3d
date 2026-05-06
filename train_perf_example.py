"""
训练循环性能分析示例

这个脚本展示了如何在训练循环中添加性能分析
"""

import time
import random

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.timers = {}
    
    def start(self, name):
        """开始计时"""
        if name not in self.timers:
            self.timers[name] = {'total': 0.0, 'count': 0}
        self.timers[name]['start'] = time.time()
    
    def end(self, name):
        """结束计时"""
        if name in self.timers and 'start' in self.timers[name]:
            elapsed = time.time() - self.timers[name]['start']
            self.timers[name]['total'] += elapsed
            self.timers[name]['count'] += 1
    
    def report(self, iteration):
        """打印性能报告"""
        print(f"\n===== 性能报告 [迭代 {iteration}] =====")
        for name, data in sorted(self.timers.items()):
            if data['count'] > 0:
                avg_ms = (data['total'] / data['count']) * 1000
                print(f"  {name:20s}: {avg_ms:6.2f} ms")

# 模拟优化器类
class MockOptimizer:
    def __init__(self, num_params):
        self.params = [{'grad': None} for _ in range(num_params)]
    
    def step(self):
        """模拟优化器步骤"""
        for p in self.params:
            if p['grad'] is None:
                continue
            # 模拟参数更新
            pass
    
    def zero_grad(self):
        """清零梯度"""
        for p in self.params:
            p['grad'] = None

# 模拟渲染函数
def mock_render(camera_region=0):
    """模拟渲染耗时"""
    time.sleep(0.02)  # 模拟20ms渲染
    return {'loss': random.random() * 0.1}

# 主训练循环
def train(num_iterations, num_optimizer_params, num_regions=65):
    analyzer = PerformanceAnalyzer()
    optimizer = MockOptimizer(num_optimizer_params)
    
    print(f"训练开始: {num_iterations} 次迭代, {num_optimizer_params} 个优化器参数, {num_regions} 个区域")
    
    for iteration in range(num_iterations):
        # 区域切换检查
        analyzer.start("区域检查")
        current_region = iteration % num_regions  # 模拟区域切换
        analyzer.end("区域检查")
        
        # 渲染
        analyzer.start("渲染")
        render_pkg = mock_render(camera_region=current_region)
        analyzer.end("渲染")
        
        # 损失计算
        analyzer.start("损失计算")
        loss = render_pkg['loss']
        analyzer.end("损失计算")
        
        # 反向传播（模拟）
        analyzer.start("反向传播")
        time.sleep(0.01)  # 模拟10ms反向传播
        analyzer.end("反向传播")
        
        # 优化器步骤
        analyzer.start("优化器步骤")
        optimizer.step()
        optimizer.zero_grad()
        analyzer.end("优化器步骤")
        
        # 每10次迭代打印报告
        if iteration % 10 == 9:
            analyzer.report(iteration + 1)
    
    print("\n训练完成!")

if __name__ == "__main__":
    # 测试不同配置
    print("="*50)
    print("配置1: 原版 (3个MLP, ~7000参数)")
    print("="*50)
    train(num_iterations=50, num_optimizer_params=7000, num_regions=1)
    
    print("\n" + "="*50)
    print("配置2: 多区域版本 (195个MLP, ~480000参数)")
    print("="*50)
    train(num_iterations=50, num_optimizer_params=480000, num_regions=65)
