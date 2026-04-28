#!/usr/bin/env python3
import argparse
import json
import sys

# 添加当前目录到路径
sys.path.insert(0, '/workspace')

from render9-huang import Renderer

def main():
    parser = argparse.ArgumentParser(description="快速渲染单个视图")
    parser.add_argument("--model_path", required=True, type=str, help="模型输出路径")
    parser.add_argument("--data_path", required=True, type=str, help="数据路径")
    parser.add_argument("--rotation_w", required=True, type=str, help="相机到世界的旋转矩阵（JSON 格式）")
    parser.add_argument("--p_w", required=True, type=str, help="相机在世界坐标系中的位置（JSON 格式）")
    parser.add_argument("--show_level", action="store_true", help="是否显示不同层级")
    parser.add_argument("--ape_code", type=int, default=None, help="外观编码（可选）")
    
    args = parser.parse_args()
    
    # 解析 JSON 参数
    rotation_w = json.loads(args.rotation_w)
    p_w = json.loads(args.p_w)
    
    print("初始化 Renderer...")
    # 初始化 Renderer
    renderer = Renderer(
        model_path=args.model_path,
        data_path=args.data_path
    )
    
    print("渲染单个视图...")
    # 渲染单个视图
    renderer.render_set_one_view(rotation_w, p_w, show_level=args.show_level, ape_code=args.ape_code)
    print("渲染完成！")

if __name__ == "__main__":
    main()
