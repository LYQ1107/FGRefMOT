import os

# 定义文件夹列表
folders = [
    "v_1LwtoLPw2TU_c014", "v_1LwtoLPw2TU_c016",
    "v_1yHWGw8DH4A_c601", "v_1yHWGw8DH4A_c609", "v_1yHWGw8DH4A_c610",
    "v_2j7kLB-vEEk_c007", "v_2j7kLB-vEEk_c009", "v_2j7kLB-vEEk_c010",
    "v_4LXTUim5anY_c010", "v_4LXTUim5anY_c012",
    "v_-6Os86HzwCs_c007", "v_-6Os86HzwCs_c009",
    "v_ApPxnw_Jffg_c015", "v_ApPxnw_Jffg_c016",
    "v_CW0mQbgYIF4_c005",
    "v_Dk3EpDDa3o0_c007",
    "v_gQNyhv8y0QY_c013",
    "v_HdiyOtliFiw_c008", "v_HdiyOtliFiw_c010"
]

# 根目录
root_dir = "/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/expression"
# 输出文件路径
output_file = "/data1/liuyeqiang/TempMam/datasets/data_path/seqmap_sports.txt"

# 打开输出文件以写入模式
with open(output_file, 'w') as f:
    # 遍历每个文件夹
    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        # 检查文件夹是否存在
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # 遍历文件夹中的所有文件
            for file in os.listdir(folder_path):
                # 检查文件是否为 JSON 文件
                if file.endswith('.json'):
                    # 去除文件扩展名
                    json_name = os.path.splitext(file)[0]
                    # 写入文件夹名和 JSON 文件名
                    line = f"{folder}+{json_name}\n"
                    f.write(line)