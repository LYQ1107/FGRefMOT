import os
import json
import argparse


def generate_opposite_json(ref_dir, json_file_path, video_name, sentence):
    """
    读取一个已有的 JSON 文件，根据参考目录中的信息，生成一个新的 JSON 文件，其中包含原文件中没有的目标 ID，
    且这些目标 ID 必须在当前帧中存在。

    :param ref_dir: 参考目录路径，包含了以帧序号命名的 txt 文件
    :param json_file_path: 输入的 JSON 文件路径
    :param video_name: 视频名称
    :param sentence: 描述语句，用于新 JSON 文件中的对应字段
    """
    # 存储每个帧中出现的目标 ID
    target_ids_per_frame = {}
    for file_name in os.listdir(ref_dir):
        if file_name.endswith('.txt'):
            frame_number = int(file_name.split('.')[0])
            file_path = os.path.join(ref_dir, file_name)
            with open(file_path, 'r') as file:
                lines = [line.strip().split() for line in file.readlines()]
                target_ids = [int(line[1]) for line in lines]
                target_ids_per_frame[frame_number] = target_ids

    with open(json_file_path, 'r') as file:
        original_data = json.load(file)

    data = {"label": {}, "ignore": {}, "video_name": video_name, "sentence": sentence}

    # 找出原 JSON 文件中未包含但在当前帧中存在的目标 ID
    for frame_number_str, target_ids in original_data["label"].items():
        frame_number = int(frame_number_str)
        if frame_number in target_ids_per_frame:
            frame_target_ids_in_ref = set(target_ids_per_frame[frame_number])
            for target_id in target_ids:
                frame_target_ids_in_ref.discard(target_id)
            if frame_number not in data["label"]:
                data["label"][frame_number] = []
            data["label"][frame_number].extend(list(frame_target_ids_in_ref))

    # 将 JSON 文件命名为 sentence 替换空格后的形式，并保存到当前工作目录下
    new_json_file_path = os.path.join(os.getcwd(), sentence.replace(" ", "-") + ".json")
    with open(new_json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate opposite JSON file.")
    parser.add_argument("--ref_dir", type=str, help="Path of the reference directory",
                        default="/data1/liuyeqiang/SportsMOT/expression/v_1LwtoLPw2TU_c006/refer")
    parser.add_argument("--json_file_path", type=str, help="Path of the input JSON file",
                        default="/data1/liuyeqiang/SportsMOT/expression/v_1LwtoLPw2TU_c006/volleyball-players-at-the-front-court.json")
    parser.add_argument("--video_name", type=str, help="Video name", default="v_1LwtoLPw2TU_c006")
    parser.add_argument("--sentence", type=str, help="Sentence description", default="male volleyball players")
    args = parser.parse_args()

    generate_opposite_json(args.ref_dir, args.json_file_path, args.video_name, args.sentence)