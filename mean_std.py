import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

def compute_mean_std(dataset_path, batch_size=64, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # 转换为[0, 1]区间的Tensor
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    print("📊 正在统计均值与标准差...")
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # [B, C, H*W]
        mean += images.mean(2).sum(0)  # 每张图片每通道求均值
        std += images.std(2).sum(0)    # 每张图片每通道求标准差
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    print(f"\n✅ 统计完毕，共处理图像: {total_images} 张")
    print(f"Mean: {mean.tolist()}")
    print(f"Std:  {std.tolist()}")

    return mean, std

# 👉 替换为你的数据集路径
dataset_path = './test_data/val'  # 或 'train' 文件夹所在路径
compute_mean_std(dataset_path)
