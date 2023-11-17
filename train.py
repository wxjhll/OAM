import time
import datetime
import numpy as np
#################utils
import utils
from utils.myloss import my_loss
from utils.dataset import get_dateloder
from utils.seed import set_seed
##############torch
import torch.nn as nn
import torch
from torchsummary import summary
#############网络
from model.get_model import load_model
#########################
def train_model(batch_size=32, epochs=100, mylr=1e-4, train_dir='', wname='',
                 size_=(128, 128), mymodel=None, split=0.95):
    train_dataloader, val_dataloader = \
        get_dateloder(train_dir, split, batch_size, size_)
    device = torch.device("cuda")
    model = mymodel
    # model.load_state_dict(torch.load('./weight/unet_m11d13.pth'))
    # model.eval()
    model = model.to(device)
    summary(model, (1, 128, 128))
    loss_fn = my_loss()
    loss_fn = loss_fn.to(device)
    learning_rate = mylr
    optimizer =torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    milest = list(range(10, epochs + 1, 10))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milest, gamma=0.6)
    # 4. 记录训练过程中的损失
    train_losses = []
    val_losses = []
    minloss = 1

    # 训练
    model.train()
    for epoch in range(epochs):
        size = len(train_dataloader) * batch_size
        train_loss = 0.0
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)

            # resiz = F.interpolate(pred, size_, mode='bilinear')
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            if batch % 10000 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"epoch: {epoch + 1} loss: {loss:>5f}  [{current:>5d}/{size:>5d}]")
            # print(loss)

        scheduler.step()
        if optimizer.param_groups[0]['lr'] < 1e-6:
            optimizer.param_groups[0]['lr'] = 1e-6

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        # if(epoch%10==0):
        #     torch.save(model.state_dict(), './weight/{}.pth'.format(epoch))
        # 评估
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch, (X, y) in enumerate(val_dataloader):
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                # resiz= F.interpolate(outputs, size_, mode='bilinear')
                loss = loss_fn(outputs, y)
                val_loss += loss.item()  # 记录验证集总损失和
            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)
        # 5.3 打印训练过程中的损失
        print('[Epoch %d]lr：%f | Train Loss: %.5f | Val Loss: %.5f' %
              (epoch + 1, optimizer.param_groups[0]['lr'], train_loss, val_loss))

        # timestamp = int(time.time())
        if(val_loss<minloss):
            minloss=val_loss
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, './checkpoint/best.pth'.format(wname))
    torch.save({'epoch': epochs,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }, './checkpoint/{}_latest.pth'.format(wname))
    loss_data = np.array([train_losses, val_losses])
    np.save('./result/{}_loss.npy'.format(wname), loss_data)
    # plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    # plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # #plt.show()
    # plt.savefig('./result1/loss_curve{}.png'.format(wname))


if __name__ == '__main__':
    set_seed(45)
    model = load_model("swin_denoise")
    train_model(batch_size=16, epochs=60, train_dir='D:/aDeskfile/train/at'
                , wname='swin_unet_m11d16',size_=(128, 128),
                mymodel=model, split=0.8, mylr=1e-4)
