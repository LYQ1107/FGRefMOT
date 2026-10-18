import os
import numpy as np


def convert_to_left_top_coordinates(source_dir, target_dir):
    """
    将源目录下以中心点坐标形式存储的txt文件内容转换为左上角坐标形式，并保存到目标目录下。

    :param source_dir: 源目录路径，包含以中心点坐标形式存储的txt文件
    :param target_dir: 目标目录路径，用于保存转换为左上角坐标后的txt文件
    """
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录下的所有子目录和文件
    for root, dirs, files in os.walk(source_dir):
        for file_name in files:
            if file_name.endswith('.txt'):
                source_file_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(source_file_path, source_dir)
                target_file_path = os.path.join(target_dir, relative_path)
                target_dir_name = os.path.dirname(target_file_path)
                if not os.path.exists(target_dir_name):
                    os.makedirs(target_dir_name)

                # 读取原文件内容
                data = np.loadtxt(source_file_path)
                converted_data = []
                for line in data:
                    category_id = int(line[0])
                    target_id = int(line[1])
                    center_x = line[2]
                    center_y = line[3]
                    width = line[4]
                    height = line[5]

                    # 计算左上角坐标
                    left_x = center_x - width / 2
                    top_y = center_y - height / 2

                    # 构建转换后的行数据
                    converted_line = np.array([category_id, target_id, left_x, top_y, width, height])
                    converted_data.append(converted_line)

                # 将转换后的数据保存到目标文件
                np.savetxt(target_file_path, np.array(converted_data), fmt='%d %d %.6f %.6f %.6f %.6f')


if __name__ == "__main__":
    source_dir = "/data/DATASETS/TRACKING/JDE/SportsMOT/sportsmot_publish/dataset/labels_with_ids/"
    target_dir = "/data1/liuyeqiang/TempMam/data/Dataset/refer-sports/KITTI/labels_with_ids/image_02"
    convert_to_left_top_coordinates(source_dir, target_dir)