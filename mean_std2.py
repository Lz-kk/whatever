from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
def compute_mean_std(dataset_path, batch_size=64, image_size=224):
    # 检查数据集路径是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径 {dataset_path} 不存在。")
        return None, None

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # 转换为[0, 1]区间的Tensor
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 初始化变量
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    print("📊 正在统计均值与标准差...")
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # [B, C, H*W]
        # 计算当前批次每个通道的均值
        mean += images.mean(2).sum(0)
        total_images += batch_samples

    # 计算整体均值
    mean /= total_images

    # 重新遍历数据集计算标准差
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        # 扩展 mean 张量以匹配 images 张量的维度
        expanded_mean = mean.unsqueeze(0).unsqueeze(2).expand_as(images)
        # 计算当前批次每个通道的标准差
        std += ((images - expanded_mean) ** 2).sum([0, 2])

    # 计算整体标准差
    std = torch.sqrt(std / (total_images * image_size * image_size))

    print(f"\n✅ 统计完毕，共处理图像: {total_images} 张")
    print(f"Mean: {mean.tolist()}")
    print(f"Std:  {std.tolist()}")

    return mean, std


def visualize_mean_std(mean, std):
    # 通道名称
    channels = ['红', '绿', '蓝']
    # 将张量转换为列表
    mean = mean.tolist()
    std = std.tolist()
    # 创建柱状图
    x_pos = np.arange(len(channels))
    bar_colors = ['red', 'green', 'blue']
    plt.bar(x_pos, mean, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10, color=bar_colors)
    for i, channel in enumerate(channels):
        plt.text(x_pos[i], -0.05, channel, ha='center', color=bar_colors[i])
    plt.ylabel('数值')
    plt.title('图像通道的均值和标准差')
    plt.ylim(0, max(mean) * 1.2)  # 调整纵轴范围
    # 显示图形
    plt.show()


# 👉 替换为你的数据集路径
dataset_path = './test_data/val'  # 或 'train' 文件夹所在路径
mean, std = compute_mean_std(dataset_path)
if mean is not None and std is not None:
    visualize_mean_std(mean, std)



