
def process_kitti_data(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # 筛选掉包含 'DontCare' 的行
    filtered_lines = [line for line in lines if 'DontCare' not in line]

    # 要删除的列索引，从0开始计数（第3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17列）
    # 注意：删除顺序从右到左，以避免索引问题
    columns_to_remove = {2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16}

    processed_lines = []

    for line in filtered_lines:
        # 将每行拆分成单个字段
        fields = line.split()

        # 删除不需要的列
        fields = [field for idx, field in enumerate(fields) if idx not in columns_to_remove]

        # 获取原始的边界框坐标
        x_min = float(fields[2])
        y_min = float(fields[3])
        x_max = float(fields[4])
        y_max = float(fields[5])

        # 计算新的边界框参数
        x_top_left = x_min
        y_top_left = y_min
        width = x_max - x_min
        height = y_max - y_min

        # 构建新的字段列表
        # 保留第0, 1列（即索引为0, 1）
        # 然后添加转换后的四个新值
        new_fields = [
            fields[0],  # 类别名称
            fields[1],  # 置信度
            f"{x_top_left:.6f}",  # 新的x坐标
            f"{y_top_left:.6f}",  # 新的y坐标
            f"{width:.6f}",       # 宽度
            f"{height:.6f}"       # 高度
        ]

        # 在字段末尾添加三个 '1'
        new_fields.extend(['1', '1', '1'])

        # 将过滤后的字段以逗号分隔符重新组合成字符串
        processed_line = ','.join(new_fields)
        processed_lines.append(processed_line + '\n')

    # 将结果写入输出文件
    with open(output_file, 'w') as outfile:
        outfile.writelines(processed_lines)

if __name__ == "__main__":
    # 输入文件名和输出文件名
    input_filename = '/data/DATASETS/TRACKING/JDE/KITTI/label/training/label_02/0020.txt'  # 替换为你的输入文件名
    output_filename = '/data/DATASETS/TRACKING/JDE/KITTI/training/image_02/0020/gt.txt'  # 替换为你的输出文件名

    # 调用函数处理文件
    process_kitti_data(input_filename, output_filename)

    print(f"处理完成！结果已保存到 {output_filename}")
