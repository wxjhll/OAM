import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from data.make_dataset import *
from model import unet1
from model.net3 import net3
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.net import UNet
import torch.nn as nn
from model.someUnet import Unet
from tqdm import tqdm
from model.se_reunet import Shallow_SeResUNet
import torch.nn.functional as F

def data(img_dir,split=0.8):
    transform = transforms.Compose([
        transforms.Resize([128,128]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])])

    train_at, train_ping, val_at, val_ping = split_train_val(imgage_dir=img_dir,
                                                             split=split)
    train_dataset = MyDataset(input_dir=train_at,
                              ground_dir=train_ping,
                              transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True
                                  , num_workers=4, drop_last=False)
    val_dataset = MyDataset(input_dir=val_at,
                            ground_dir=val_ping,
                            transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False
                                , num_workers=4, drop_last=False)
    return train_dataloader,val_dataloader

if __name__ == '__main__':
    # 加载模型和测试数据集
    net =Shallow_SeResUNet(n_channels=1, n_classes=1, deep_supervision = False,
                            dropout = False, rate = 0.1)
    net.load_state_dict(torch.load('./weight/best{}.pth'))
    train_loader, test_loader=data(img_dir='D:/aDeskfile/slm/at',split=0.95)

    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()

    # 定义损失函数和准确率计算器
    criterion = nn.MSELoss()
    total_loss = 0.0

    # 在测试集上进行循环
    smaple_loss=[]
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            # 将图像和标签移动到 GPU 上（如果有的话）
            images, labels = images.to(device), labels.to(device)

            # 前向传播计算预测结果
            outputs = net(images)
            #outputs = F.interpolate(outputs, [512,512], mode='bilinear')
            # 计算损失和准确率
            loss = criterion(outputs, labels)
           #print(loss.item())
            smaple_loss.append(loss.item())

        # plt.plot(range(1, len(smaple_loss) + 1), smaple_loss, label='Training Loss')
        # plt.show()
            #累加损失和准确率
            total_loss += loss.item() * images.size(0)
            mae=torch.abs(outputs.cpu().squeeze()-labels[0].cpu().squeeze())
            print('ping_var:',torch.var(labels[0].cpu().squeeze()),'compention_var:',torch.var(mae))
            plt.subplot(221)
            plt.imshow(labels[0].cpu().squeeze(), cmap='jet')
            plt.clim(0,1)
            plt.title('true')
            plt.subplot(222)
            plt.imshow(outputs[0].cpu().squeeze(), cmap='jet')
            plt.clim(0, 1)
            plt.title('pred')
            plt.subplot(223)
            plt.imshow(images[0].cpu().squeeze(), cmap='jet')
            plt.title('at')
            plt.subplot(224)
            plt.imshow(mae, cmap='jet')
            plt.clim(0, 1)
            plt.title('mae')
            plt.show()

    # 将模型移动到设备上

    # 调用测试函数进行测试
    #test_loss = test(net, test_loader, device)

    # 打印测试结果
    #print('Test Loss: {:.5f}'.format(test_loss))
