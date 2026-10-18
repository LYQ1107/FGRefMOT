import matplotlib.pyplot as plt
import os

# 数据
sequence_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
41, 42
                    ]
counts = [286, 250, 544, 295, 725, 537, 1197, 552, 500, 535,
          826, 1176, 1197, 726, 1825, 651, 876, 702, 526, 875, 726,825,403,550,500,235,250,575,246,352,383,465,243,272,239,442,700,1325,875,775,1025,549]

# 设置全局字体为新罗马字体
plt.rcParams['font.family'] = 'serif'  # 用系统默认衬线字体


# 绘制柱形图
plt.figure(figsize=(12,5))
plt.bar(sequence_lengths, counts, width=1.0, color='#FFEBDC', edgecolor='black')  # 使用十六进制颜色

# 坐标轴设置
plt.xlabel('Sequence length', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim([0, 45])

plt.tight_layout()

# 指定目录路径
output_dir = '/data1/liuyeqiang/TempMam/zhifangtu'
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, 'sequence_length_histogram.png')

# 保存图片
plt.savefig(file_path, dpi=300)
plt.show()
plt.close()
