import os


def delete_files_and_dirs(root_dir):
    """
    递归删除指定根目录下的所有文件和子目录。

    :param root_dir: 需要删除文件和目录的根目录路径
    """
    for root, dirs, files in os.walk(root_dir, topdown=False):
        # 删除所有文件
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                os.remove(file_path)
                print(f"已删除文件: {file_path}")
            except OSError as e:
                print(f"删除文件 {file_path} 时出错: {e}")

        # 删除所有子目录
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                os.rmdir(dir_path)
                print(f"已删除目录: {dir_path}")
            except OSError as e:
                print(f"删除目录 {dir_path} 时出错: {e}")

    # 最后删除根目录（如果根目录为空）
    try:
        os.rmdir(root_dir)
        print(f"已删除根目录: {root_dir}")
    except OSError as e:
        print(f"删除根目录 {root_dir} 时出错: {e}")


if __name__ == "__main__":
    root_dir = "/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/expression"
    delete_files_and_dirs(root_dir)