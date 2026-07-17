import os
import numpy as np
import torch
from torch import nn
import torch.optim as optim

from torchvision import transforms, models, datasets
import imageio
import time
import warnings
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('TkAgg')

warnings.filterwarnings("ignore")
import random
import sys
import copy
import json
from PIL import Image

data_dir = './data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/val'
test_dir = data_dir + '/test'

model_name='resnet18'
filename = 'best.pt'
feature_extract = True

data_transforms = {
    'train':
        transforms.Compose([
            transforms.Resize([96, 96]),  #输入HW
            transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
            transforms.CenterCrop(64),  # 从中心开始裁剪
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
            transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
            # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
            transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
        ]),
    'val':
        transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'test':
        transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}


batch_size = 8
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                  ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
               ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

with open('cat_to_name.json', 'r', encoding='utf-8') as f:
    cat_to_name = json.load(f)



# 是否用GPU训练
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_func = getattr(models, model_name)
    model_ft = model_func(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, num_classes)
    )

    return model_ft

model_ft = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU还是CPU计算
model_ft = model_ft.to(device)


# 是否训练所有层
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# 优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)  # 要训练啥参数，你来定
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)  # 学习率每10个epoch衰减成原来的1/10
criterion = nn.CrossEntropyLoss()


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, filename='best.pt'):
    # 咱们要算时间的
    since = time.time()
    # 也要记录最好的那一次
    best_acc = 0
    # 模型也得放到你的CPU或者GPU
    model.to(device)
    # 训练过程中打印一堆损失和指标
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    # 学习率
    LRs = [optimizer.param_groups[0]['lr']]
    # 最好的那次模型，后续会变的，先初始化
    best_model_wts = copy.deepcopy(model.state_dict())
    # 一个个epoch来遍历
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # 放到你的CPU或GPU
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                # 训练阶段更新权重
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)  # 0表示batch那个维度
                running_corrects += torch.sum(preds == labels.data)  # 预测结果最大的和真实值是否一致

            epoch_loss = running_loss / len(dataloaders[phase].dataset)  # 算平均
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since  # 一个epoch我浪费了多少时间
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),  # 字典里key就是各层的名字，值就是训练好的权重
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                # scheduler.step(epoch_loss)#学习率衰减
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()
        scheduler.step()  # 学习率衰减

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果,等着一会测试
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


def fig_out(val_acc_history, train_acc_history, valid_losses, train_losses,name):
    epochs = range(1, len(train_acc_history) + 1)
    # 创建一个 1x2 的子图布局
    fig, axes = plt.subplots(1, 2, figsize=(12, 10))

    # 绘制训练和验证准确率曲线
    axes[0].plot(epochs, train_acc_history, 'b-', label='Training Accuracy')
    axes[0].plot(epochs, val_acc_history, 'r-', label='Validation Accuracy')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    # 绘制训练和验证损失曲线
    axes[1].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[1].plot(epochs, valid_losses, 'r-', label='Validation Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    # 调整子图之间的间距
    plt.tight_layout()
    plt.savefig(f'{name}.png')
    # 显示图形
    plt.show()


model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,
                                                                                            criterion, optimizer_ft,
                                                                                            num_epochs=100)
fig_out(val_acc_history, train_acc_history, valid_losses, train_losses,'train_first')

for param in model_ft.parameters():
    param.requires_grad = True

# 再继续训练所有的参数，学习率调小一点
optimizer = optim.Adam(model_ft.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 加载之前训练好的权重参数

checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer, num_epochs=10,)

fig_out(val_acc_history, train_acc_history, valid_losses, train_losses,'train_second')

# 加载模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

# 得到一个batch的测试数据
dataiter = iter(dataloaders['test'])
images, labels = next(dataiter)

model_ft.eval()

if train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)

_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())


def im_convert(tensor):
    """ 展示数据"""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


fig = plt.figure(figsize=(20, 20))
columns = 4
rows = 2

for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(labels[idx].item())]),
                 color=("green" if cat_to_name[str(preds[idx])] == cat_to_name[str(labels[idx].item())] else "red"))
plt.show()
