import os


def sort_txt_by_column(file_path):
    lines = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split(', ')
            lines.append(parts)

    # 按照第二列（索引为1）转换为整数后进行排序
    sorted_lines = sorted(lines, key=lambda x: int(x[1]))

    with open(file_path, 'w') as file:
        for sorted_line in sorted_lines:
            file.write(', '.join(sorted_line) + '\n')


# 指定包含多个gt.txt文件的目录路径
directory_path = "/data1/liuyeqiang/SportsMOT/sportsmot_publish/dataset/train"
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file == "gt.txt":
            file_path = os.path.join(root, file)
            sort_txt_by_column(file_path)