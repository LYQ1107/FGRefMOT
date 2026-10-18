import os
import shutil


def remove_specific_folders(root_path, targets=None):
    """
    在 root_path 下递归查找并删除名为 targets 中任一名称的子文件夹。
    :param root_path: 根目录
    :param targets: 需要删除的子文件夹名称列表
    """
    if targets is None:
        targets = ["refer", "new"]  # 默认要删除的文件夹

    for current_dir, dirnames, filenames in os.walk(root_path):
        # 注意: 在遍历过程中直接修改 dirnames 可能导致 walk 出错，这里先记录需要删除的文件夹
        to_remove = []

        for d in dirnames:
            if d in targets:
                to_remove.append(d)

        # 实际删除
        for d in to_remove:
            folder_path = os.path.join(current_dir, d)
            if os.path.isdir(folder_path):
                print(f"Deleting folder: {folder_path}")
                shutil.rmtree(folder_path)
                # 从dirnames列表中移除，以防后续重复遍历
                dirnames.remove(d)


if __name__ == "__main__":
    root_dir = "/data1/liuyeqiang/SportsMOT/expression"

    # 调用函数，删除名为 "refer" 和 "new" 的文件夹
    remove_specific_folders(root_dir, ["refer", "new"])
