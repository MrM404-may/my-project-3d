#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
from os import makedirs
import torch
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from render9_class import Render9Huang
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

def main():
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Render9Huang testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=10, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show_level", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # 初始化系统状态 (RNG)
    safe_state(args.quiet)

    # 创建Render9Huang实例
    renderer = Render9Huang(
        model_path=args.model_path,
        source_path=getattr(args, 'source_path', ''),
        iteration=args.iteration,
        resolution_scales=args.resolution_scales,
        quiet=args.quiet
    )
    
    # 加载场景和高斯模型
    renderer.load_scene_and_gaussians()
    
    # 渲染训练集和测试集
    renderer.render_sets(
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        show_level=args.show_level,
        ape_code=args.ape
    )

if __name__ == "__main__":
    main()
