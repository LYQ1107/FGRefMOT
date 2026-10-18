import os

# 定义路径
source_dir = '/data1/liuyeqiang/RMOT/data/Dataset/refer-kitti/KITTI/labels_with_ids/image_02/0001'
target_dir = '/data1/liuyeqiang/RMOT/data/Dataset/refer-kitti/KITTI/labels_with_ids/fix/0001'
reference_dir = '/data/DATASETS/TRACKING/JDE/KITTI/training/labels_with_ids/train/0001'

# 定义ID映射字典（反转）
id_mapping = {
    14: 823, 11: 820, 12: 821, 13: 822, 1: 825, 2: 824, 3: 826,
    16: 915, 15: 917, 17: 914, 18: 913, 5: 827, 4: 829, 8: 831,
    9: 833, 6: 830, 20: 828, 7: 834, 21: 832, 23: 837, 22: 838,
    28: 909, 26: 848, 27: 847, 25: 836, 34: 851, 30: 850, 33: 849,
    36: 843, 38: 845, 37: 844, 39: 854, 24: 839, 29: 840, 31: 841,
    35: 842, 40: 853, 41: 852, 42: 855, 43: 856, 44: 857, 46: 860,
    48: 916, 49: 858, 45: 859, 47: 861, 50: 862, 51: 863, 52: 864,
    53: 865, 54: 866, 55: 867, 56: 868, 57: 869, 58: 872, 59: 874,
    60: 871, 61: 875, 62: 911, 63: 877, 67: 873, 70: 880, 65: 876,
    69: 878, 64: 883, 66: 879, 71: 882, 72: 886, 73: 881, 75: 885,
    74: 884, 76: 887, 77: 895, 78: 888, 79: 889, 80: 890, 81: 892,
    82: 891, 83: 893, 84: 894, 85: 896, 86: 897, 89: 900, 88: 899,
    87: 898, 81: 902, 90: 901, 95: 905, 94: 904, 92: 906, 93: 903,
    96: 908
}

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 遍历源目录下的所有txt文件
for filename in os.listdir(source_dir):
    if filename.endswith('.txt'):
        source_file_path = os.path.join(source_dir, filename)
        target_file_path = os.path.join(target_dir, filename)
        reference_file_path = os.path.join(reference_dir, filename)

        # 读取源文件
        with open(source_file_path, 'r') as source_file:
            source_lines = source_file.readlines()

        # 读取参考文件（如果存在）
        reference_data = {}
        if os.path.exists(reference_file_path):
            with open(reference_file_path, 'r') as reference_file:
                reference_lines = reference_file.readlines()
                for ref_line in reference_lines:
                    parts = ref_line.strip().split()
                    if len(parts) < 6:
                        continue
                    ref_id = int(parts[1])
                    reference_data[ref_id] = parts[2:6]

        # 修正源文件内容
        corrected_lines = []
        for line in source_lines:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            target_id = int(parts[1])
            if target_id in id_mapping:
                ref_id = id_mapping[target_id]
                if ref_id in reference_data:
                    corrected_line = f"{parts[0]} {target_id} " + " ".join(reference_data[ref_id]) + "\n"
                    corrected_lines.append(corrected_line)
                else:
                    corrected_lines.append(line)
            else:
                corrected_lines.append(line)

        # 将修正后的内容写入目标文件
        with open(target_file_path, 'w') as target_file:
            target_file.writelines(corrected_lines)

        print(f"Processed {filename}")
