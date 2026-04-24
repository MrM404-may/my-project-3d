import struct
import numpy as np

# ====================== 请修改这里 ======================
PLY_PATH = r"/root/autodl-tmp/Octree-GS/Octree-GS/output/Huang-region/merged_anchors.ply"  # 你的ply文件路径
# ========================================================

# 固定按照你给的头文件顺序定义属性（必须和你提供的头完全一致）
properties = [
    "x", "y", "z",
    "level", "extra_level", "region", "info",
    *[f"f_offset_{i}" for i in range(30)],
    *[f"f_anchor_feat_{i}" for i in range(32)],
    "opacity",
    *[f"scale_{i}" for i in range(6)],
    *[f"rot_{i}" for i in range(4)],
]

# 每个属性都是 float，4 字节
FLOAT_SIZE = 4
VERTEX_SIZE = len(properties) * FLOAT_SIZE

# 找到 region 字段的索引
region_idx = properties.index("region")

# 读取 PLY
with open(PLY_PATH, "rb") as f:
    # 读取到 end_header 为止
    header = []
    while True:
        line = f.readline().decode("ascii", errors="ignore").strip()
        header.append(line)
        if line == "end_header":
            break

    # 读取顶点数量（从头文件解析）
    vertex_count = None
    for line in header:
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
            break

    print(f"顶点总数：{vertex_count}")
    print(f"每个顶点字节数：{VERTEX_SIZE}")
    print(f"正在读取 region 字段...")

    # 批量读取二进制数据（速度极快）
    data = f.read(vertex_count * VERTEX_SIZE)

    # 只提取 region 字段
    regions = []
    for i in range(vertex_count):
        offset = i * VERTEX_SIZE + region_idx * FLOAT_SIZE
        region = struct.unpack_from("<f", data, offset)[0]
        regions.append(region)

# 统计唯一值
unique_regions = sorted(np.unique(regions))
print("\n===== region 字段的所有唯一值 =====")
print(unique_regions)
print(f"\n共有 {len(unique_regions)} 种不同的 region 值")