import os
import json
import argparse


def generate_json(ref_dir, video_name, sentence):
    """
    根据给定的参考目录、视频名称和描述语句，将参考目录下文本文件中的数据处理并转换为指定格式的JSON数据，最后保存为JSON文件，
    其中"label"字典的键（代表图片序号）将按照从小到大的顺序排列。

    :param ref_dir: 参考目录路径，包含了以帧序号命名的txt文件
    :param video_name: 视频名称
    :param sentence: 描述语句，用于JSON文件中的对应字段
    """
    data = {"label": {}, "ignore": {}, "video_name": video_name, "sentence": sentence}

    # 遍历ref_dir目录下的所有txt文件
    for file_name in os.listdir(ref_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(ref_dir, file_name)
            frame_number = int(file_name.split('.')[0])  # 获取帧序号（文件名去除后缀后转换为整数）

            # 读取txt文件内容并进行处理
            with open(file_path, 'r') as file:
                lines = [line.strip().split() for line in file.readlines()]
                for line in lines:
                    target_id = int(line[1])  # 获取目标ID
                    if frame_number not in data["label"]:
                        data["label"][frame_number] = []
                    data["label"][frame_number].append(target_id)

    # 对每个键对应列表中的值进行去重处理
    for key in data["label"]:
        data["label"][key] = list(set(data["label"][key]))

    # 对"label"字典按照键（图片序号）从小到大排序
    sorted_label = dict(sorted(data["label"].items(), key=lambda x: int(x[0])))
    data["label"] = sorted_label

    # 将JSON文件命名为sentence替换空格后的形式，并保存到当前工作目录下
    json_file_path = os.path.join(os.getcwd(), sentence.replace(" ", "-") + ".json")
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and generate JSON file.")
    parser.add_argument("--ref_dir", type=str, help="Path of the reference directory",
                        default="/data1/liuyeqiang/SportsMOT/expression/v_1LwtoLPw2TU_c006/refer")
    parser.add_argument("--video_name", type=str, help="Video name", default="v_1LwtoLPw2TU_c006")
    parser.add_argument("--sentence", type=str, help="Sentence description", default="male volleyball players")
    args = parser.parse_args()

    generate_json(args.ref_dir, args.video_name, args.sentence)