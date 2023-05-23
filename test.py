import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from data.make_dataset import *
from model import unet1
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.net import UNet
import torch.nn as nn


def data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.16436058], std=[0.1736191]),

    ])

    inputdir2 = get_image_paths('D:/Ldata/NOAM/val/AT')
    grounddir2 = get_image_paths('D:/Ldata/NOAM/val/ping')
    val_dataset = MyDataset(input_dir=inputdir2,
                              ground_dir=grounddir2,
                              transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True
                                  , num_workers=2, drop_last=False)
    return val_dataloader
# 定义测试函数
def test(net, test_loader, device):
    # 将模型设置为评估模式
    net.eval()

    # 定义损失函数和准确率计算器
    criterion = nn.MSELoss()
    total_loss = 0.0

    # 在测试集上进行循环
    with torch.no_grad():
        for images, labels in test_loader:
            # 将图像和标签移动到 GPU 上（如果有的话）
            images, labels = images.to(device), labels.to(device)

            # 前向传播计算预测结果
            outputs = net(images)

            # 计算损失和准确率
            loss = criterion(outputs, labels)

            # 累加损失和准确率
            total_loss += loss.item() * images.size(0)

    # 计算平均损失和准确率
    mean_loss = total_loss / len(test_loader.dataset)

    return mean_loss

# 测试代码示例
if __name__ == '__main__':
    # 加载模型和测试数据集
    net =UNet()
    net.load_state_dict(torch.load('model.pth'))
    test_loader=data()

    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()

    # 定义损失函数和准确率计算器
    criterion = nn.MSELoss()
    total_loss = 0.0

    # 在测试集上进行循环
    with torch.no_grad():
        for images, labels in test_loader:
            # 将图像和标签移动到 GPU 上（如果有的话）
            images, labels = images.to(device), labels.to(device)

            # 前向传播计算预测结果
            outputs = net(images)

            # 计算损失和准确率
            loss = criterion(outputs, labels)

            # 累加损失和准确率
            total_loss += loss.item() * images.size(0)
            plt.subplot(121)
            plt.imshow(labels[0].cpu().squeeze(), cmap='gray')
            plt.clim(0,1)
            plt.subplot(122)
            plt.imshow(outputs[0].cpu().squeeze(), cmap='gray')
            plt.clim(0, 1)
            plt.show()

    # 将模型移动到设备上





    # 调用测试函数进行测试
    #test_loss = test(net, test_loader, device)

    # 打印测试结果
    #print('Test Loss: {:.5f}'.format(test_loss))
