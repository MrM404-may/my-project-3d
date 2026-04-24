import os

# ====================== 请在这里修改你的文件夹路径 ======================
IMAGE_FOLDER = r"/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma3w/images-kk"  # Windows 示例
# IMAGE_FOLDER = "/home/user/Pictures"                  # Linux/macOS 示例
# ======================================================================

# 支持的图片格式（可自行增删）
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

def rename_images():
    # 获取文件夹里所有文件
    files = os.listdir(IMAGE_FOLDER)
    # 只保留图片文件
    image_files = [f for f in files if f.lower().endswith(IMAGE_EXTS)]
    
    if not image_files:
        print("未找到任何图片文件")
        return

    # 按文件名排序（保证顺序稳定）
    image_files.sort()

    count = 1
    for filename in image_files:
        # 旧路径
        old_path = os.path.join(IMAGE_FOLDER, filename)
        # 新文件名：5位数字 + .png
        new_name = f"{count:05d}.png"
        new_path = os.path.join(IMAGE_FOLDER, new_name)

        # 重命名
        os.rename(old_path, new_path)
        print(f"已重命名: {filename} -> {new_name}")
        count += 1

    print(f"\n完成！共重命名 {count-1} 张图片")

if __name__ == "__main__":
    # 重要提醒
    print("⚠️  运行前请备份图片，避免误操作覆盖！")
    confirm = input("确定要继续重命名吗？(y/n): ")
    if confirm.lower() == 'y':
        rename_images()
    else:
        print("已取消")