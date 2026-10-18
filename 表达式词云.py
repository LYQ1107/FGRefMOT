import os
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 顶层 expression 文件夹路径
root_dir = '/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/expression'

# 用于拼接所有句子
all_sentences = []

# 遍历所有子文件夹和 JSON 文件，提取 sentence 字段
for subdir, _, files in os.walk(root_dir):
    for filename in files:
        if filename.endswith('.json'):
            json_path = os.path.join(subdir, filename)

            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                sentence = data.get("sentence", "")
                all_sentences.append(sentence)

            except Exception as e:
                print(f"读取失败: {json_path} 错误信息: {e}")

custom_words = ['moving', 'holding', 'for alling', 'free Man','running','shooting','goalkeepers','three-point line']

# 拼接为一个长文本
text = ' '.join(all_sentences + custom_words * 10)

# 生成更紧凑、词更多的词云
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    colormap='summer',         # 推荐色彩丰富方案
    max_words=750,            # 展示更多词
    min_font_size=5,          # 最小字体更小
    font_step=1,              # 字体缩放更细致
    prefer_horizontal=0.7,    # 更倾向于横向展示
    margin=1,                 # 更小间距
    collocations=False
).generate(text)

# 显示和保存词云图
plt.figure(figsize=(12, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()

# 保存路径
output_path = '/data1/liuyeqiang/TempMam/keshihua/biaodashi_ciyun/wordcloud_compact.png'
plt.savefig(output_path, dpi=300)
plt.show()
