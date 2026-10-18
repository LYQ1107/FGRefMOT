import os
import json

# 顶层 expression 文件夹路径
expression_root = '/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/expression'
# KITTI labels 路径根目录
kitti_root = '/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/KITTI/labels_with_ids/image_02'

# 输出文件路径
output_txt = '/data1/liuyeqiang/TempMam/keshihua/biaodashi/output_labels5.txt'

ratios = []

# 收集所有子文件夹（按名称排序）
subdirs = sorted([os.path.join(expression_root, d) for d in os.listdir(expression_root)
                  if os.path.isdir(os.path.join(expression_root, d))])

for subdir in subdirs:
    video_name = os.path.basename(subdir)

    # 构造对应 KITTI 路径
    kitti_dir = os.path.join(kitti_root, video_name)

    if not os.path.exists(kitti_dir):
        print(f"警告：找不到对应的 KITTI 路径 {kitti_dir}，跳过")
        continue

    # 获取该视频的 frame 数（.txt 文件数量）
    count_base = len([f for f in os.listdir(kitti_dir) if f.endswith('.txt')])

    if count_base == 0:
        print(f"警告：{kitti_dir} 中无有效 .txt 文件，跳过")
        continue

    # 遍历该 expression 子目录下所有 .json 文件
    for filename in sorted(os.listdir(subdir)):
        if filename.endswith('.json'):
            json_path = os.path.join(subdir, filename)

            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                label_dict = data.get("label", {})
                num_keys = len(label_dict)

                ratio = num_keys / count_base
                ratios.append(ratio)

            except Exception as e:
                print(f"读取失败: {json_path} 错误信息: {e}")

# 保存为列表格式
with open(output_txt, 'w') as f:
    f.write(str(ratios))

print(f"共处理 {len(ratios)} 个 JSON 文件，比例结果已写入 {output_txt}")
