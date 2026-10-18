import matplotlib.pyplot as plt
import numpy as np

# 数据标签
datasets = ['Refer-Sports', 'Refer-KITTI', 'GroOT', 'Refer-Dance']

# 各区间比例（堆叠数据）
values = [
    [0.0, 0.3799, 0.0292, 0.3799, 0.0552, 0.0162, 0.1396],
    [0, 0.4389, 0.3215, 0.12, 0.05, 0.03, 0.036],
    [0, 0.998, 0.002, 0, 0, 0, 0],
    [0.9486, 0.02, 0.015, 0.01, 0.004, 0.002, 0.0004]
]

# 区间颜色和标签
colors = ['#EEB7B4', '#FFE699', '#E5F2DD', '#c6dbef', '#dadaeb', '#ccebc5', '#6387CF']
labels = ['0', '0-0.2', '0.2-0.4', '0.4-0.6', '0.4-0.7', '0.6-0.7', '0.6-0.8']

# 绘图设置
fig, ax = plt.subplots(figsize=(7, 4))  # 更紧凑的高度
bar_width = 0.7  # 更宽的条形
y = np.arange(len(datasets))
left = np.zeros(len(datasets))

# 堆叠条形图
for idx in range(len(values[0])):
    bar = [v[idx] for v in values]
    ax.barh(y, bar, left=left, height=bar_width, color=colors[idx], label=labels[idx])
    left += bar

# 百分比标注
# for i, row in enumerate(values):
#     if row[0] > 0:
#         ax.text(row[0] / 2, i, f'{row[0]*100:.2f}%', va='center', ha='center', fontsize=9, color='white')
#     if row[1] > 0:
#         ax.text(row[0] + row[1]/2, i, f'{row[1]*100:.2f}%', va='center', ha='center', fontsize=9, color='black')

# y轴标签
ax.set_yticks(y)
ax.set_yticklabels(datasets, fontsize=10)

# 坐标轴设置
ax.set_xlim(0, 1)
ax.set_xticks(np.linspace(0, 1, 6))
ax.set_xticklabels([f'{int(x*100)}%' for x in np.linspace(0, 1, 6)], fontsize=10)
ax.invert_yaxis()

# 隐藏上右边框，保留左下边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 坐标轴刻度
ax.tick_params(axis='y', which='both', length=4, width=1)
ax.tick_params(axis='x', which='both', length=4, width=1)

# 图例
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=7, fontsize=8)

# 箭头
ax.annotate('', xy=(1.02, -0.5), xytext=(1, -0.5), arrowprops=dict(facecolor='black', arrowstyle='->'))

# 布局调整
# 布局调整
plt.tight_layout(rect=[0, 0.05, 1, 1])

# 保存图像
output_path = '/data1/liuyeqiang/TempMam/keshihua/xiangxingtu/stacked_bar.png'
plt.savefig(output_path, dpi=300)

# 显示图像
plt.show()

