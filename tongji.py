import os


def count_json_files(root_dir):
    """递归统计根目录下所有子文件夹中的JSON文件数量"""
    json_count = 0
    json_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 过滤出所有.json文件
        for filename in filenames:
            if filename.endswith('.json'):
                json_count += 1
                json_files.append(os.path.join(dirpath, filename))

    return json_count, json_files


# 指定根目录
root_directory = '/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/expression'

# 执行统计
total_count, file_list = count_json_files(root_directory)

# 输出结果
print(f"总JSON文件数量: {total_count}")
print("\nJSON文件列表:")
for file_path in file_list:
    print(file_path)