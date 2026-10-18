import os
import json

# 顶层 expression 文件夹路径
root_dir = '/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/expression'
output_txt = '/data1/liuyeqiang/TempMam/keshihua/biaodashi/output_unique_label_values_normalized.txt'

# KITTI 路径
kitti_root = '/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/KITTI/labels_with_ids/image_02'

normalized_counts = []

# 遍历所有子文件夹和文件
for subdir, _, files in os.walk(root_dir):
    if not files:
        continue

    # 提取当前视频文件夹名
    video_folder_name = os.path.basename(subdir)
    kitti_video_path = os.path.join(kitti_root, video_folder_name)

    # 统计该视频文件夹中所有 txt 文件的最大行数
    max_lines = 0
    if os.path.isdir(kitti_video_path):
        for txt_file in os.listdir(kitti_video_path):
            if txt_file.endswith('.txt'):
                txt_path = os.path.join(kitti_video_path, txt_file)
                try:
                    with open(txt_path, 'r') as f:
                        lines = f.readlines()
                        max_lines = max(max_lines, len(lines))
                except Exception as e:
                    print(f"读取失败: {txt_path} 错误信息: {e}")
    else:
        print(f"未找到对应的 KITTI 路径: {kitti_video_path}")
        continue

    # 遍历当前 expression 文件夹下所有 json
    for filename in files:
        if filename.endswith('.json'):
            json_path = os.path.join(subdir, filename)

            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                label_dict = data.get("label", {})
                unique_values = set()

                for key in label_dict:
                    values = label_dict[key]
                    unique_values.update(values)

                # 防止除以0
                if max_lines > 0:
                    normalized_ratio = len(unique_values) / max_lines
                else:
                    normalized_ratio = 0.0

                normalized_counts.append(normalized_ratio)

            except Exception as e:
                print(f"读取失败: {json_path} 错误信息: {e}")

# 写入归一化后的结果
with open(output_txt, 'w') as f:
    f.write(str(normalized_counts))

print(f"共处理 {len(normalized_counts)} 个 JSON 文件，已保存归一化 label 值比例到 {output_txt}")
