import os
import shutil

# 设置路径
base_dir = 'voc2007'
labels_dir = base_dir + '/labels'      # 存放YOLO标签的目录
images_dir = base_dir + '/images'      # 原始图片目录
Set = base_dir + '/ImageSets/'
output_root = 'test_data'


# 划分文件路径
split_files = {
    'train': 'train.txt',
    'val': 'val.txt',
    'test': 'test.txt'
}

# 支持图片扩展名
image_exts = ['.jpg', '.jpeg', '.png']

# 读取划分数据
splits = {}
for split_name, filepath in split_files.items():
    with open(Set + filepath, 'r', encoding='utf-8') as f:
        print(f)
        print(split_name)
        splits[split_name] = set()  # 初始化每个划分集为一个空集合
        for line in f.readlines():
            print(line.strip())
            splits[split_name].add(line.strip())  # 将每行数据加入集合

# 遍历所有标签文件
for label_file in os.listdir(labels_dir):
    if not label_file.endswith('.txt'):
        continue

    label_path = os.path.join(labels_dir, label_file)
    with open(label_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        continue  # 空标签跳过

    # 获取第一个标签作为类别
    cls = lines[0].split()[0]

    # 找到对应的图片
    base_name = os.path.splitext(label_file)[0]
    image_path = None
    image_ext = None
    for ext in image_exts:
        img_candidate = os.path.join(images_dir, base_name + ext)
        if os.path.exists(img_candidate):
            image_path = img_candidate
            image_ext = ext
            break

    if not image_path:
        print(f"⚠️ 未找到图片：{base_name}")
        continue

    # 确定该图片属于哪个划分集
    split_target = None
    for split_name, name_set in splits.items():
        if base_name in name_set:
            split_target = split_name
            break

    if not split_target:
        print(f"⚠️ {base_name} 不在任何划分中")
        continue

    # 创建目标文件夹路径
    img_dst_dir = os.path.join(output_root, split_target, cls)
    os.makedirs(img_dst_dir, exist_ok=True)

    # 复制图片和标签到相应目录
    shutil.copy(image_path, os.path.join(img_dst_dir, base_name + image_ext))
    shutil.copy(label_path, os.path.join(img_dst_dir, label_file))

print("✅ 图片和标签按划分集和类别组织完成！")
