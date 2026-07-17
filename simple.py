"""
用于单张测试
"""
import os
import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
from torch import nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import json
import torch
from PIL import Image
import torchvision.transforms as transforms
from tkinter import filedialog
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

# 定义图片转换（应与你训练/测试时的 transform 保持一致）
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "efficientnet_b3":
        weights = EfficientNet_B3_Weights.DEFAULT if use_pretrained else None
        model_ft = efficientnet_b3(weights=weights)
        # set_parameter_requires_grad(model_ft, feature_extract)
        in_features = model_ft.classifier[1].in_features
        model_ft.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )
        return model_ft
    else:
        raise ValueError(f"Unsupported model_name {model_name}")

model_name = "efficientnet_b3"
feature_extract = True
model_ft = initialize_model(model_name, 102, feature_extract, use_pretrained=True)  #num——classs 测试前先检查分类，保持一致

train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")
model_ft.to(device)
with open('cat_to_name.json', 'r', encoding='utf-8') as f:
    cat_to_name = json.load(f)
filename = 'best_efficientnet.pt'#模型名

checkpoint = torch.load(filename)
model_ft.load_state_dict(checkpoint['state_dict'])

# 加载并预处理单张图片
def process_single_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加 batch 维度
    return image.to(device)

# 预测单张图片类别
def predict_single_image(image_path):
    image_tensor = process_single_image(image_path)
    model_ft.eval()
    with torch.no_grad():
        output = model_ft(image_tensor)
        _, pred = torch.max(output, 1)
        pred_class = pred.item()
        return pred_class
# 弹出窗口选择图片
def choose_image():
    root = tk.Tk()
    root.withdraw()  # 不显示主窗口
    file_path = filedialog.askopenfilename(
        title='选择图片文件',
        filetypes=[('Image Files', '*.jpg *.jpeg *.png *.bmp *.webp')]
    )
    return file_path
# 可视化单张图片预测
def visualize_single_image(image_path, true_label=None):
    image_tensor = process_single_image(image_path)
    pred_class = predict_single_image(image_path)
    pred_name = cat_to_name[str(pred_class)]

    # 从文件夹名获取真实标签
    true_label_name = None
    if true_label is None:
        true_label_name = os.path.basename(os.path.dirname(image_path))
        true_label_name=cat_to_name[str(true_label_name)]
    else:
        true_label_name = cat_to_name[str(true_label)]

    # 判断是否预测正确
    if true_label_name is not None:
        correct = (pred_name == true_label_name)
        title_color = 'green' if correct else 'red'
        title_text = f"Predicted: {pred_name} (True: {true_label_name})"
    else:
        title_color = 'black'
        title_text = f"Predicted: {pred_name}"

    # 显示图片
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title(title_text, fontsize=16, color=title_color)
    plt.axis('off')
    plt.show()

# 执行测试
image_path = choose_image()
if image_path:
    visualize_single_image(image_path)
else:
    print("未选择图片。")
