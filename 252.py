import os
import shutil

# 原始文件的路径
source_file = "/data1/liuyeqiang/RMOT/data/Dataset/refer-kitti/KITTI/labels_with_ids/image_02/0013/000132.txt"

# 目标文件夹路径
destination_folder = "/data1/liuyeqiang/RMOT/data/Dataset/refer-kitti/KITTI/labels_with_ids/image_02/0013/"

# 开始编号
start_index = 133
# 结束编号
end_index = 206

# 检查原始文件是否存在
if not os.path.isfile(source_file):
    print(f"Source file does not exist: {source_file}")
else:
    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)

    for i in range(start_index, end_index + 1):
        # 创建目标文件路径
        destination_file = os.path.join(destination_folder, f"{i:06}.txt")

        # 复制文件
        shutil.copy(source_file, destination_file)
        print(f"Copied to {destination_file}")

print("All files copied.")
