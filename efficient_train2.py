import os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
import time
import warnings
import matplotlib
import matplotlib.pyplot as plt
import random
import copy
import json
from PIL import Image
from tqdm import tqdm
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('TkAgg')
warnings.filterwarnings("ignore")

# 路径设置
data_dir = './data/'
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

model_name = 'efficientnet_b5'
filename = 'best_efficientnet2.pt'#模型名
feature_extract = True

# 数据增强和预处理，EfficientNet-B5通常输入大小为240x240
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([240, 240]),
        transforms.RandomRotation(30),
        transforms.CenterCrop(240),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5050356984138489, 0.5589249134063721, 0.3754347264766693],
                             [0.1842256486415863, 0.17761452496051788, 0.1822347790002823])
    ]),
    'val': transforms.Compose([
        transforms.Resize([240, 240]),
        transforms.ToTensor(),
        transforms.Normalize([0.4963628947734833, 0.5522289276123047, 0.3652009963989258],
                             [0.19102221727371216, 0.18155233561992645, 0.18533922731876373])
    ]),
    'test': transforms.Compose([
        transforms.Resize([240, 240]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

batch_size = 32


def load_data(data_dir, data_transforms, batch_size):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['train', 'val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
                   ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names


dataloaders, dataset_sizes, class_names = load_data(data_dir, data_transforms, batch_size)

with open('cat_to_name.json', 'r', encoding='utf-8') as f:
    cat_to_name = json.load(f)

train_on_gpu = torch.cuda.is_available()
logging.info('Training on GPU' if train_on_gpu else 'Training on CPU')
device = torch.device("cuda:0" if train_on_gpu else "cpu")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        # 先冻结所有层
        for param in model.parameters():
            param.requires_grad = False

        # 解冻最后两到三层 block
        for name, param in model.named_parameters():
            if "features.6" in name or "features.7" in name:
                param.requires_grad = True


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "efficientnet_b5":
        weights = EfficientNet_B5_Weights.DEFAULT if use_pretrained else None
        model_ft = efficientnet_b5(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        # 这里把in_features获取到
        in_features = model_ft.classifier[1].in_features
        # 修改全连接，输出用num_classes
        model_ft.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )
        return model_ft
    else:
        raise ValueError(f"Unsupported model_name {model_name}")


model_ft = initialize_model(model_name, 102, feature_extract, use_pretrained=True)
model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
if feature_extract:
    params_to_update = [param for param in model_ft.parameters() if param.requires_grad]
optimizer_ft = optim.Adam(params_to_update, lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=20)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, filename='best.pt', patience=3):
    since = time.time()
    best_acc = 0
    model.to(device)
    val_acc_history, train_acc_history = [], []
    train_losses, valid_losses, LRs = [], [], [optimizer.param_groups[0]['lr']]
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch}/{num_epochs - 1}\n' + '-' * 10)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0

            with tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch}', unit='batch') as tepoch:
                for inputs, labels in tepoch:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, filename)
                early_stopping_counter = 0
            elif phase == 'val':
                early_stopping_counter += 1

            if phase == 'val':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        logging.info(f'Optimizer learning rate: {optimizer.param_groups[0]["lr"]:.7f}')
        LRs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

        time_elapsed = time.time() - since
        logging.info(f'time elapsed {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')

        if early_stopping_counter >= patience:
            logging.info("Early stopping triggered.")
            break

    logging.info(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


def fig_out(val_acc_history, train_acc_history, valid_losses, train_losses, name):
    train_acc_history = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in train_acc_history]
    val_acc_history = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in val_acc_history]
    train_losses = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in train_losses]
    valid_losses = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in valid_losses]

    epochs = range(1, len(train_acc_history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_acc_history, 'b-', label='Train Acc')
    axes[0].plot(epochs, val_acc_history, 'r-', label='Val Acc')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[1].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[1].plot(epochs, valid_losses, 'r-', label='Val Loss')
    axes[1].set_title('Loss')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'{name}.png')
    plt.show()


# 初始训练
# 是否训练所有层
params_to_update = model_ft.parameters()
logging.info("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            logging.info(f"\t {name}")
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            logging.info(f"\t {name}")

# 微调训练
# model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(
#     model_ft, dataloaders, criterion, optimizer_ft, scheduler, num_epochs=10, filename=filename)
# fig_out(val_acc_history, train_acc_history, valid_losses, train_losses, 'train_first')

# 解冻全部参数微调
for param in model_ft.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model_ft.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
checkpoint = torch.load(filename)
model_ft.load_state_dict(checkpoint['state_dict'])
# 全参训练
# model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(
#     model_ft, dataloaders, criterion, optimizer, scheduler, num_epochs=10)
# fig_out(val_acc_history, train_acc_history, valid_losses, train_losses, 'train_second')

# 测试可视化
dataiter = iter(dataloaders['test'])
images, labels = next(dataiter)
model_ft.eval()
output = model_ft(images.to(device))
_, preds_tensor = torch.max(output, 1)
preds = preds_tensor.cpu().numpy()


def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    return image.clip(0, 1)


fig = plt.figure(figsize=(20, 10))
# 调整子图布局
plt.subplots_adjust(hspace=0.3, wspace=0.2)
for idx in range(8):
    ax = fig.add_subplot(2, 4, idx + 1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    pred_name = cat_to_name[str(preds[idx])]
    label_name = cat_to_name[str(labels[idx].item())]
    # 设置标题字体大小和样式
    ax.set_title(f"{pred_name} ({label_name})", color=("green" if pred_name == label_name else "red"),
                 fontsize=12, fontweight='bold')
    # 添加子图边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)

plt.tight_layout()
plt.show()

