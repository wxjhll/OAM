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



def getdataloader(batch_size=32,transform=None):
    inputdir = get_image_paths('D:/Ldata/NOAM/AT')
    grounddir = get_image_paths('D:/Ldata/NOAM/ping')
    train_dataset = MyDataset(input_dir=inputdir,
                              ground_dir=grounddir,
                              transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True
                                  ,num_workers=2,drop_last=False)
    return train_dataloader
def train_model(batch_size = 16,epochs = 100):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.16436058], std=[0.1736191]),


    ])

    inputdir = get_image_paths('D:/Ldata/NOAM/AT')
    grounddir = get_image_paths('D:/Ldata/NOAM/ping')
    train_dataset = MyDataset(input_dir=inputdir,
                              ground_dir=grounddir,
                              transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True
                                  , num_workers=2, drop_last=False)
    #验证集
    inputdir2 = get_image_paths('D:/Ldata/NOAM/val/AT')
    grounddir2 = get_image_paths('D:/Ldata/NOAM/val/ping')
    val_dataset = MyDataset(input_dir=inputdir2,
                              ground_dir=grounddir2,
                              transform=transform)
    val_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True
                                  , num_workers=2, drop_last=False)

    learning_rate = 1e-4
    device = torch.device("cuda")

    model=UNet()
    #model.load_state_dict(torch.load('model.pth'))
    model = model.to(device)
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 4. 记录训练过程中的损失
    train_losses = []
    val_losses = []
    #训练
    model.train()
    for epoch in range(epochs):
        size = len(train_dataloader)*batch_size
        train_loss = 0.0
        val_loss = 0.0
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item() * X.size(0)
            if batch % 20 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"epoch: {epoch} loss: {loss:>5f}  [{current:>5d}/{size:>5d}]")
            #print(loss)
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)
        #评估
        model.eval()
        with torch.no_grad():
            for batch, (X, y)in enumerate(val_dataloader):
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss =  loss_fn(outputs, y)
                val_loss += loss.item() * X.size(0)  # 记录验证集总损失和
            val_loss /= len(val_dataset)
            val_losses.append(val_loss)
        # 5.3 打印训练过程中的损失
        print('[Epoch %d] Train avgLoss: %.5f | Val avgLoss: %.5f' % (epoch + 1, train_loss, val_loss))
    # 7. 保存权重文件
    torch.save(model.state_dict(), './weight/model.pth')
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    train_model()
