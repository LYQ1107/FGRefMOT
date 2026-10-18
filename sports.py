import os
import json
import argparse


def compare_txt_files(new_dir, ref_dir, output_path):
    new_files = os.listdir(new_dir)
    ref_files = os.listdir(ref_dir)
    missing_lines = []
    for file_name in set(new_files).intersection(ref_files):
        new_file_path = os.path.join(new_dir, file_name)
        ref_file_path = os.path.join(ref_dir, file_name)

        new_lines = []
        with open(new_file_path, 'r') as new_file:
            new_lines = [line.strip() for line in new_file.readlines()]

        ref_lines = []
        with open(ref_file_path, 'r') as ref_file:
            ref_lines = [line.strip() for line in ref_file.readlines()]

        for index, ref_line in enumerate(ref_lines):
            is_missing = True
            for new_line in new_lines:
                ref_line_parts = ref_line.split()
                new_line_parts = new_line.split()
                ref_line_data = [
                    "{:.6f}".format(float(x)) for x in ref_line_parts[2:]]
                new_line_data = [
                    "{:.6f}".format(float(x)) for x in new_line_parts[2:]]
                if ref_line_parts[:2] == new_line_parts[:2]:  # and ref_line_data == new_line_data:
                    is_missing = False
                    break
            if is_missing:
                missing_lines.append(f"{file_name.split('.')[0]} {ref_line}\n")

    with open(output_path, 'w') as output_file:
        output_file.writelines(missing_lines)


def generate_json(txt_file_path, json_file_path, video_name, sentence):
    data = {"label": {}, "ignore": {}, "video_name": video_name, "sentence": sentence}
    id_dict = {}
    sorted_lines = []
    # 先读取txt文件内容并进行排序
    with open(txt_file_path, 'r') as file:
        lines = [line.strip().split() for line in file.readlines()]
        # 按照第三列数值排序，第三列相同的再按照第一列数值从小到大排序
        sorted_lines = sorted(lines, key=lambda x: (int(x[2]), int(x[0])))

    # 记录每个第三列数值对应的索引范围，用于后续填充data["label"]字典
    index_ranges = {}
    prev_id = None
    start_index = None
    for index, line in enumerate(sorted_lines):
        current_id = int(line[2])
        if prev_id is None:
            prev_id = current_id
            start_index = index
        elif current_id == prev_id:
            if prev_id not in index_ranges:
                index_ranges[prev_id] = []
            index_ranges[prev_id].append((start_index, index))
            start_index = index + 1  # 更新起始索引，指向下一组的开始位置
        else:
            if prev_id is not None:
                if prev_id not in index_ranges:
                    index_ranges[prev_id] = []
                index_ranges[prev_id].append((start_index, index - 1))
            prev_id = current_id
            start_index = index

    # 根据索引范围填充id_dict字典
    for third_column_id, index_ranges_list in index_ranges.items():
        id_dict[third_column_id] = []
        for min_index, max_index in index_ranges_list:
            for i in range(min_index, max_index + 1):
                index = int(sorted_lines[i][0])
                id_dict[third_column_id].append(index)

    # 对每个third_column_id对应的index_list按两个一组进行处理，填充data["label"]字典
    for third_column_id, index_list in id_dict.items():
        length = len(index_list)
        if length > 2:
            for i in range(0, length, 2):
                sub_list = index_list[i:i + 2]
                min_index = min(sub_list)
                max_index = max(sub_list)
                for j in range(min_index, max_index + 1):
                    if j not in data["label"]:
                        data["label"][j] = []
                    data["label"][j].append(third_column_id)
        else:
            min_index = min(index_list)
            max_index = max(index_list)
            for i in range(min_index, max_index + 1):
                if i not in data["label"]:
                    data["label"][i] = []
                data["label"][i].append(third_column_id)

    # 对每个键对应列表中的值进行去重处理
    for key in data["label"]:
        data["label"][key] = list(set(data["label"][key]))

    # 将data["label"]字典的键按照从小到大排序，生成新的有序字典
    sorted_label = dict(sorted(data["label"].items(), key=lambda x: int(x[0])))
    data["label"] = sorted_label

    # 检查并处理label中的键值对，根据对应txt文件中是否存在对应的值来决定是否删除键值对
    refer_dir = os.path.join(os.path.dirname(txt_file_path), "refer")
    for key, value_list in data["label"].items():
        file_name = f"{int(key):06}.txt"
        file_path = os.path.join(refer_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                lines = [line.strip().split()[1] for line in file.readlines()]
                new_value_list = []
                for v in value_list:
                    if str(v) in lines:
                        new_value_list.append(v)
                data["label"][key] = new_value_list
        else:
            print(f"Warning: File {file_path} does not exist.")

    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--new_dir", type=str, help="Path of the new directory",default="/data1/liuyeqiang/SportsMOT/expression/v_1yHWGw8DH4A_c029/new")
    parser.add_argument("--ref_dir", type=str, help="Path of the reference directory",default="/data1/liuyeqiang/SportsMOT/expression/v_1yHWGw8DH4A_c029/refer")
    parser.add_argument("--video_name", type=str, help="Video name",default="v_1yHWGw8DH4A_c029")
    parser.add_argument("--sentence", type=str, help="Sentence description",default="football players holding the ball")
    args = parser.parse_args()

    output_txt_path = os.path.join(os.getcwd(), args.sentence.replace(" ", "-") + ".txt")
    json_file_path = os.path.join(os.getcwd(), args.sentence.replace(" ", "-") + ".json")
    compare_txt_files(args.new_dir, args.ref_dir, output_txt_path)
    generate_json(output_txt_path, json_file_path, args.video_name, args.sentence)
