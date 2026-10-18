import os

# 定义根目录
root_dir = "/data1/liuyeqiang/TempMam/exps/default_rk/results_epoch99"

# 遍历根目录下的所有文件夹
for root, dirs, files in os.walk(root_dir):
    # 检查是否存在 gt.txt 文件
    if "gt.txt" in files:
        gt_file_path = os.path.join(root, "gt.txt")
        gt_new_file_path = os.path.join(root, "gt.txt")

        # 打开 gt.txt 文件进行读取
        with open(gt_file_path, 'r') as gt_file:
            lines = gt_file.readlines()

        # 处理每一行数据
        new_lines = []
        for line in lines:
            # 分割每行数据
            parts = line.strip().split(',')
            # 将第一列的数减一
            parts[0] = str(int(parts[0]) - 1)
            # 重新组合数据
            new_line = ','.join(parts) + '\n'
            new_lines.append(new_line)

        # 打开 gt_new.txt 文件进行写入
        with open(gt_new_file_path, 'w') as gt_new_file:
            gt_new_file.writelines(new_lines)

        print(f"已处理 {gt_file_path}，生成 {gt_new_file_path}")