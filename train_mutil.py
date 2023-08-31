import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from data.make_mutiloam import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.Aencoder import *
import torch.nn as nn
from torchsummary import summary
from model.net3 import *
from utils.myloss import my_loss
def getdataloader(batch_size=32,transform=None):
    inputdir = get_image_paths('/data/home/Deepin/mutil/noat/')
    grounddir = get_image_paths('/data/home/Deepin/mutil/noat/')
    train_dataset = MyDataset(input_dir=inputdir,
                              ground_dir=grounddir,
                              transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True
                                  ,num_workers=4,drop_last=False)
    return train_dataloader
def train_model(batch_size = 32,epochs = 100):

    transform = transforms.Compose([transforms.Resize([64,64]),transforms.ToTensor()])

    train_at, train_ping, val_at, val_ping = split_train_val(imgage_dir='/data/home/Deepin/mutil/at/'
                                                             , split=0.9)
    train_dataset = MyDataset(input_dir=train_at,
                              ground_dir=train_ping,
                              transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True
                                  , num_workers=4, drop_last=False)
    #验证集
    val_dataset = MyDataset(input_dir=val_at,
                              ground_dir=val_ping,
                              transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False
                                  , num_workers=4, drop_last=False)

    device = torch.device("cuda")

    model=net3()
    #model.load_state_dict(torch.load('model.pth'))
    model = model.to(device)
    summary(model, (1,64,64))
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    milest = np.linspace(10, epochs, 10)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milest, gamma=0.5)
    # 4. 记录训练过程中的损失
    train_losses = []
    val_losses = []
    minloss=1
    #训练
    model.train()
    for epoch in range(epochs):
        size = len(train_dataloader)*batch_size
        train_loss = 0.0
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item() * X.size(0)
            if batch % 5 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"epoch: {epoch+1}/{epochs} loss: {loss:>5f}  [{current:>5d}/{size:>5d}]")
            #print(loss)
        print("第%d个epoch的学习率：%f" % (epoch+1, optimizer.param_groups[0]['lr']))
        #scheduler.step()
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)
        #评估
        model.eval()
        val_loss = 0.0
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
        if(val_loss<minloss):
            minloss=val_loss
            torch.save(model.state_dict(),'./weight/mutiloam_best.pth')
    # 7. 保存权重文件
    torch.save(model.state_dict(), './weight/mutiloam_model.pth')
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #plt.show()
    timestamp = int(time.time())
    plt.savefig('./result/loss_curve{}.png'.format(timestamp))



if __name__ == '__main__':
    train_model(batch_size = 16,epochs = 100)
