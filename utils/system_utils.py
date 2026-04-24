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

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    import os
    if not os.path.exists(folder):
        # 如果目录不存在，返回0
        return 0
    saved_iters = []
    for fname in os.listdir(folder):
        try:
            iter_num = int(fname.split("_")[-1])
            saved_iters.append(iter_num)
        except ValueError:
            # 忽略非数字的文件名，如'checkpoints'
            pass
    if saved_iters:
        return max(saved_iters)
    else:
        # 如果没有找到有效的迭代次数，返回0
        return 0
