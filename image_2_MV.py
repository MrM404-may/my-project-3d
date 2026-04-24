import cv2
import os

# ===================== 【请修改这里的参数】 =====================
image_folder = "/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma0422/images"  # 你的图片文件夹路径
video_name = "output_video.mp4"  # 输出视频文件名
fps = 15  # 视频帧率（越大播放越快，常用 15/24/30）
# =================================================================

# 获取文件夹里所有图片，并按文件名排序（00000, 00001...）
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()  # 关键：保证图片顺序正确

# 读取第一张图片，获取宽高
first_img = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_img.shape

# 定义视频编码器并创建视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 格式
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# 逐张写入视频
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# 释放资源
video.release()
cv2.destroyAllWindows()

print(f"✅ 视频生成完成！保存为：{video_name}")

# import os
# import cv2

# # ===================== 【请修改这里的参数】 =====================
# base_render_path = "/root/autodl-tmp/Octree-GS/Octree-GS/output/Huang-region/train/ours_480000/renders"
# fps = 30  # 视频帧率
# start_region = 0  # 起始 region
# end_region = 15   # 结束 region（改成你实际的最大 region）
# # =================================================================

# # 循环遍历每个 region 文件夹
# for camera_region in range(start_region, end_region + 1):
#     # 拼接当前 region 图片文件夹路径
#     image_folder = os.path.join(base_render_path, str(camera_region))
#     # 输出视频名
#     video_name = f"output_video_{camera_region}.mp4"
    
#     print(f"正在处理 region {camera_region}：{image_folder}")

#     # 获取文件夹里所有图片，并按文件名排序
#     images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
#     images.sort()

#     # 读取第一张图片获取宽高
#     first_img = cv2.imread(os.path.join(image_folder, images[0]))
#     height, width, layers = first_img.shape

#     # 定义视频编码器
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

#     # 逐张写入视频
#     for image in images:
#         img_path = os.path.join(image_folder, image)
#         frame = cv2.imread(img_path)
#         video.write(frame)

#     # 释放当前视频资源
#     video.release()

#     print(f"✅ region {camera_region} 视频完成：{video_name}\n")

# cv2.destroyAllWindows()
# print("🎉 所有 region 视频全部生成完成！")