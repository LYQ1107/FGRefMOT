import os


def main():
    # 要遍历的根目录（里面包含 image_02 文件夹）
    search_root = "/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/KITTI/training/image_02"

    # 用来作为相对路径的起始点（从这里开始截取）
    # 即我们想让输出结果从 "KITTI/training/..." 开始
    base_for_relpath = "/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti"

    # 输出文件路径
    output_file = "/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/KITTI/training/refer-kitti.train"

    # 用来存储所有文件的相对路径
    rel_paths = []

    for root, dirs, files in os.walk(search_root):
        for filename in files:
            # 如果只想收集.jpg文件，可以加一个判断:
            if filename.lower().endswith(".jpg"):
                abs_path = os.path.join(root, filename)  # 文件的绝对路径
                # 获得相对于 base_for_relpath 的路径
                # 这样就能得到 "KITTI/training/image_02/xxx/yyy.jpg" 这样的结果
                rel_path = os.path.relpath(abs_path, start=base_for_relpath)
                rel_paths.append(rel_path)

    # 如果希望按字符串排序，可以在这里进行排序
    rel_paths.sort()

    # 将结果写入输出文件
    with open(output_file, "w") as f:
        for p in rel_paths:
            f.write(p + "\n")

    print(f"写入完成，共 {len(rel_paths)} 条记录。输出文件：{output_file}")


if __name__ == "__main__":
    main()
