import os
import json

# 顶层 expression 文件夹路径
root_dir = '/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/expression'

# 输出文件路径
output_txt = '/data1/liuyeqiang/TempMam/keshihua/biaodashi/output_unique_label_values.txt'

unique_counts = []

# 遍历所有子文件夹和文件
for subdir, _, files in os.walk(root_dir):
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
                    unique_values.update(values)  # 添加所有值（自动去重）

                unique_counts.append(len(unique_values))  # 记录每个文件中独特 value 的数量

            except Exception as e:
                print(f"读取失败: {json_path} 错误信息: {e}")

# 写入为 [4, 6, 3, ...] 格式
with open(output_txt, 'w') as f:
    f.write(str(unique_counts))

print(f"共处理 {len(unique_counts)} 个 JSON 文件，独特 label 值数量已写入 {output_txt}")
