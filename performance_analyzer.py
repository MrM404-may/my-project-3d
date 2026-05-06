import time
import torch

class PerformanceAnalyzer:
    """
    性能分析器：用于测量训练过程中各个步骤的耗时
    """
    
    def __init__(self):
        self.timers = {}
        self.counters = {}
        self.start_time = None
    
    def start_timer(self, name):
        """开始计时"""
        if name not in self.timers:
            self.timers[name] = {'total': 0.0, 'count': 0}
        self.timers[name]['start'] = time.time()
    
    def end_timer(self, name):
        """结束计时"""
        if name in self.timers and 'start' in self.timers[name]:
            elapsed = time.time() - self.timers[name]['start']
            self.timers[name]['total'] += elapsed
            self.timers[name]['count'] += 1
            del self.timers[name]['start']
    
    def increment_counter(self, name, value=1):
        """增加计数器"""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
    
    def print_report(self, iteration=None):
        """打印性能报告"""
        if iteration is not None:
            print(f"\n===== 性能报告 [迭代 {iteration}] =====")
        else:
            print("\n===== 性能报告 =====")
        
        # 打印计时器结果
        print("\n【计时结果】")
        for name, data in sorted(self.timers.items()):
            if data['count'] > 0:
                avg_time = data['total'] / data['count'] * 1000  # 转换为毫秒
                print(f"  {name}: {avg_time:.2f} ms/次 (共 {data['count']} 次)")
        
        # 打印计数器结果
        if self.counters:
            print("\n【计数器结果】")
            for name, value in sorted(self.counters.items()):
                print(f"  {name}: {value}")
    
    def reset(self):
        """重置所有计时器和计数器"""
        self.timers = {}
        self.counters = {}

# 使用示例
if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    
    # 模拟训练循环
    for i in range(10):
        # 模拟渲染
        analyzer.start_timer("渲染")
        time.sleep(0.01)  # 模拟渲染耗时
        analyzer.end_timer("渲染")
        
        # 模拟反向传播
        analyzer.start_timer("反向传播")
        time.sleep(0.005)  # 模拟反向传播耗时
        analyzer.end_timer("反向传播")
        
        # 模拟优化器步骤
        analyzer.start_timer("优化器")
        time.sleep(0.002)  # 模拟优化器耗时
        analyzer.end_timer("优化器")
    
    analyzer.print_report()
