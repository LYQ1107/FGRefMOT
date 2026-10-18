import os


def main():
    # 1. 读取 expression 目录下所有子文件夹的名称
    expression_root = "/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/expression"
    # 只保留目录，忽略普通文件
    expression_folders = [
        d for d in os.listdir(expression_root)
        if os.path.isdir(os.path.join(expression_root, d))
    ]

    # 转成 set 便于快速查找
    expression_set = set(expression_folders)

    # 2. 读入原列表文件 refer-kitti.train
    train_list = "/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/KITTI/training/refer-kitti.train"

    # 输出文件（过滤后）
    filtered_list = "/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/KITTI/training/refer-kitti_filtered.train"

    valid_lines = []

    with open(train_list, "r") as f:
        lines = f.readlines()

    # 3. 检查每行中对应的目录是否在 expression_set 中
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 假设路径格式为:
        # KITTI/training/image_02/v_iIxMOsCGH58_c013/000450.jpg
        # 分割后得到数组 [ "KITTI", "training", "image_02", "v_iIxMOsCGH58_c013", "000450.jpg" ]
        parts = line.split('/')

        # 安全起见，先判断长度是否足够
        if len(parts) < 4:
            # 如果行格式异常，就直接跳过或根据需求处理
            continue

        folder_name = parts[3]  # 第 4 个元素就是类似 v_iIxMOsCGH58_c013

        # 判断这个目录是否在 expression 目录下存在
        if folder_name in expression_set:
            valid_lines.append(line)
        # else：说明这个文件夹并不在 expression 文件夹里 -> 不保留

    # 4. 将保留的行写到输出文件
    with open(filtered_list, "w") as f:
        for vl in valid_lines:
            f.write(vl + "\n")

    print(f"原文件共 {len(lines)} 行，过滤后保留 {len(valid_lines)} 行。")
    print(f"过滤后结果已写入: {filtered_list}")


if __name__ == "__main__":
    main()
