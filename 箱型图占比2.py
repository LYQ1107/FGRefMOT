import ast

# 输入路径
input_txt = '/data1/liuyeqiang/TempMam/keshihua/biaodashi/output_unique_label_values_normalized.txt'

# 读取数据
with open(input_txt, 'r') as f:
    data = ast.literal_eval(f.read())  # 读取成 list

# 初始化区间统计
bins = ['0', '0-0.2', '0.2-0.4', '0.4-0.6', '0.4-0.7', '0.6-0.7', '0.6-0.8']
counts = [0] * len(bins)

# 遍历每个数值，判断属于哪个区间
for value in data:
    if value == 0:
        counts[0] += 1
    elif 0 < value <= 0.2:
        counts[1] += 1
    elif 0.2 < value <= 0.4:
        counts[2] += 1
    elif 0.4 < value <= 0.6:
        counts[3] += 1
    elif 0.6 < value <= 0.7:
        counts[4] += 1
    elif 0.7 < value <= 0.8:
        counts[5] += 1
    elif 0.8 < value <= 1.0:
        counts[6] += 1

# 转换为比例
total = sum(counts)
ratios = [round(c / total, 4) for c in counts]

print("各区间比例为：")
print(ratios)
