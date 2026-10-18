import os


def main():
    # 原文件路径
    input_file = "/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/KITTI/training/refer-kitti-full.train"

    # 过滤后生成的新文件（如果想直接覆盖原文件，可将此改为 input_file）
    output_file = "/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti/KITTI/training/refer-kitti-full_filtered.train"

    # 需要删除行中包含的关键字列表
    remove_keywords = [
        "v_1LwtoLPw2TU_c014","v_1LwtoLPw2TU_c016",
        "v_1yHWGw8DH4A_c601","v_1yHWGw8DH4A_c609","v_1yHWGw8DH4A_c610",
        "v_2j7kLB-vEEk_c007","v_2j7kLB-vEEk_c009","v_2j7kLB-vEEk_c010",
        "v_4LXTUim5anY_c010","v_4LXTUim5anY_c012",
        "v_-6Os86HzwCs_c007","v_-6Os86HzwCs_c009",
        "v_ApPxnw_Jffg_c015","v_ApPxnw_Jffg_c016",
        "v_CW0mQbgYIF4_c005",
        "v_Dk3EpDDa3o0_c007",
        "v_gQNyhv8y0QY_c013",
        "v_HdiyOtliFiw_c008","v_HdiyOtliFiw_c010",
    ]

    # 读取原文件所有行
    with open(input_file, "r") as f:
        lines = f.readlines()

    # 过滤行：只保留那些不包含任意 remove_keywords 的行
    kept_lines = []
    for line in lines:
        # 如果行包含任意一个关键词，则跳过；否则保留
        if any(keyword in line for keyword in remove_keywords):
            continue
        else:
            kept_lines.append(line)

    # 将结果写入新文件
    with open(output_file, "w") as f:
        f.writelines(kept_lines)

    print(f"原文件共有 {len(lines)} 行，过滤后剩余 {len(kept_lines)} 行。")
    print(f"已写入: {output_file}")


if __name__ == "__main__":
    main()
