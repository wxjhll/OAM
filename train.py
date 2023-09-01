import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import datetime
from data.make_dataset import *
from model import unet1
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.swin_unet import SwinUnet
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from model.se_reunet import Shallow_SeResUNet
from unet import TransUnet
from utils.myloss import my_loss
from model.someUnet  import Unet

def getdataloader(batch_size=32,transform=None):
    inputdir = get_image_paths('D:/Ldata/NOAM/AT')
    grounddir = get_image_paths('D:/Ldata/NOAM/ping')
    train_dataset = MyDataset(input_dir=inputdir,
                              ground_dir=grounddir,
                              transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True
                                  ,num_workers=4,drop_last=False)
    return train_dataloader
def train_model(batch_size = 32,epochs = 100,mylr=1e-4,train_dir='',wname='',
                size=(128,128),size_=(128,128),mymodel=Unet,split=0.95):

    transform = transforms.Compose([
       transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])])#mean=[0.15167561], std=[0.21450585]

    train_at, train_ping, val_at, val_ping = split_train_val(imgage_dir=train_dir, split=split)
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

    model=mymodel
    #model.load_state_dict(torch.load('./weight/modelslm1693414975.pth'))
    model = model.to(device)
    summary(model,(1,128,128))
    loss_fn =my_loss()
    loss_fn = loss_fn.to(device)
    learning_rate = mylr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    milest =list(range(10, epochs+1, 10))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milest, gamma=0.6)
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

            #resiz = F.interpolate(pred, size_, mode='bilinear')
            loss = loss_fn(pred , y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() * X.size(0)
            if batch % 10000 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"epoch: {epoch+1} loss: {loss:>5f}  [{current:>5d}/{size:>5d}]")
            #print(loss)

        scheduler.step()
        if optimizer.param_groups[0]['lr'] < 1e-6:
            optimizer.param_groups[0]['lr'] = 1e-6

        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        #评估
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch, (X, y)in enumerate(val_dataloader):
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                #resiz= F.interpolate(outputs, size_, mode='bilinear')
                loss =  loss_fn(outputs, y)
                val_loss += loss.item() * X.size(0)  # 记录验证集总损失和
            val_loss /= len(val_dataset)
            val_losses.append(val_loss)
        # 5.3 打印训练过程中的损失
        print('[Epoch %d]lr：%f | Train Loss: %.5f | Val Loss: %.5f' %
              (epoch + 1,optimizer.param_groups[0]['lr'], train_loss, val_loss))

        #timestamp = int(time.time())
        if(val_loss<minloss):
            minloss=val_loss
            torch.save(model.state_dict(),'./weight/best.pth')
    torch.save(model.state_dict(), './weight/modelslm{}.pth'.format(wname))

    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #plt.show()
    plt.savefig('./result/loss_curve{}.png'.format(wname))

if __name__ == '__main__':
    swin_unet = SwinUnet(embed_dim=96,
                         patch_height=4,
                         patch_width=4,
                         class_num=1)
    model=Shallow_SeResUNet(n_channels=1, n_classes=1, deep_supervision = False,
                            dropout = False, rate = 0.3)

    train_model(batch_size = 16, epochs = 60,    train_dir='D:/aDeskfile/oam_m/at'
            ,wname='b_256_',    size=[128,128],  size_=(128,128),
            mymodel= swin_unet,     split=0.95,      mylr=1e-4   )


