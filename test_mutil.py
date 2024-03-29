import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from data.make_mutiloam import *
from model import unet1
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.Aencoder import *
import torch.nn as nn
from model.net3 import *

def data():
    transform = transforms.Compose([
        transforms.Resize([64,64]),
        transforms.ToTensor(),])

    train_at, train_ping, val_at, val_ping = split_train_val(imgage_dir='/data/home/Deepin/mutil/at',
                                                             split=0.9)
    train_dataset = MyDataset(input_dir=train_at,
                              ground_dir=train_ping,
                              transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True
                                  , num_workers=4, drop_last=False)
    val_dataset = MyDataset(input_dir=val_at,
                            ground_dir=val_ping,
                            transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False
                                , num_workers=4, drop_last=False)
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
    net =net3()
    net.load_state_dict(torch.load('./weight/mutiloam_best.pth'))
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
            print(loss.item())

            # 累加损失和准确率
            total_loss += loss.item() * images.size(0)
            mae=torch.abs(outputs.cpu().squeeze()-labels[0].cpu().squeeze())
            print('ping_var:',torch.var(labels[0].cpu().squeeze()),'compention_var:',torch.var(mae))
            plt.subplot(131)
            plt.imshow(outputs[0].cpu().squeeze(), cmap='jet')
            plt.title('pred')

            plt.subplot(132)
            plt.imshow(labels[0].cpu().squeeze(), cmap='jet')
            plt.title('true')

            plt.subplot(133)
            plt.imshow(images[0].cpu().squeeze(), cmap='jet')
            plt.title('at')
            plt.show()

    # 将模型移动到设备上

    # 调用测试函数进行测试
    #test_loss = test(net, test_loader, device)

    # 打印测试结果
    #print('Test Loss: {:.5f}'.format(test_loss))
