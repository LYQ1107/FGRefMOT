import random
import ast

# 原始输出路径
output_txt = '/data1/liuyeqiang/TempMam/keshihua/biaodashi/output_labels89.txt'

# 读取原有列表
with open(output_txt, 'r') as f:
    content = f.read()
    original_counts = ast.literal_eval(content)  # 安全地将字符串解析为列表

# 目标长度
target_length = 492
current_length = len(original_counts)

# 拓展逻辑
if current_length < target_length:
    extra = random.choices(original_counts, k=target_length - current_length)
    extended_counts = original_counts + extra
else:
    extended_counts = original_counts[:target_length]  # 如果原本超过，截断

# 写回文件
with open(output_txt, 'w') as f:
    f.write(str(extended_counts))

print(f"原始数量：{current_length}，已扩展为：{len(extended_counts)}，结果写入 {output_txt}")
