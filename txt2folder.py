import os
import shutil
from collections import defaultdict

# 设置主目录路径
base_dir = "ip102_v1.1"
image_dir = os.path.join(base_dir, "images")
output_base = "laitingdedata"



txt_files = ["train.txt", "val.txt", "test.txt"]

for txt_file in txt_files:
    split_name = txt_file.split(".")[0]  # train, val, test
    txt_path = os.path.join(base_dir, txt_file)

    # 初始化类别计数器
    class_counter = defaultdict(int)

    with open(txt_path, "r") as f:
        for line in f:
            img_name, label = line.strip().split()



            src_img_path = os.path.join(image_dir, img_name)
            dst_dir = os.path.join(output_base, split_name, label)
            dst_img_path = os.path.join(dst_dir, img_name)

            os.makedirs(dst_dir, exist_ok=True)

            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)
                class_counter[label] += 1
            else:
                print(f"未找到图片: {src_img_path}")


