from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from data.make_dataset import *
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from model.get_model import load_model
def data(img_dir,split=0.8,size_=[128,128]):
    transform = transforms.Compose([
        transforms.Resize(size_),
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

def test(net,img_dir='' ,weight_path='',isplt=True,splitpoint=0.95,size=[128,128]):
    net.load_state_dict(torch.load(weight_path))
    # for name, param in net.named_parameters():
    #     print(f'Layer: {name}, Size: {param.size()}, Values: {param}')
    train_loader, test_loader = data(img_dir=img_dir, split=splitpoint,size_=size)
    device=torch.device('cuda')
    net.to('cuda')
    # 定义损失函数和准确率计算器
    criterion = nn.MSELoss()
    total_loss = 0.0
    net.eval()
    smaple_loss = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # outputs = F.interpolate(outputs, [320,320], mode='bilinear')

            loss = criterion(outputs, labels)

            # print(loss.item())
            smaple_loss.append(loss.item())

            # plt.plot(range(1, len(smaple_loss) + 1), smaple_loss, label='Training Loss')
            # plt.show()
            total_loss += loss.item() * images.size(0)
            mae = torch.abs(outputs.cpu().squeeze() - labels[0].cpu().squeeze())
            print('ping_var:', torch.var(labels[0].cpu().squeeze()),
                  'compention_var:', torch.var(mae))
            plt.clf()
            plt.subplot(221)
            plt.imshow(labels[0].cpu().squeeze(), cmap='jet')
            plt.colorbar()
            plt.clim(0, 1)
            plt.title('true')
            plt.subplot(222)
            plt.imshow(outputs[0].cpu().squeeze(), cmap='jet')
            plt.colorbar()
            plt.clim(0, 1)
            plt.title('pred')
            plt.subplot(223)
            plt.imshow(images[0].cpu().squeeze(), cmap='hot')
            plt.title('at')
            plt.subplot(224)
            plt.imshow(mae, cmap='jet')
            plt.colorbar()
            plt.clim(0, 1)
            plt.title('mae')
            plt.show(block=True)
            #plt.pause(0.1)  # 2秒钟（根据需要调整）

            # 关闭当前图形窗口
            #plt.close()




if __name__ == '__main__':
    model = load_model("swin_denoise")
    test(net=model, img_dir='D:/aDeskfile/train/at',
         weight_path='./weight/swin_unet_m11d16.pth', isplt=True,splitpoint=0.8,size=[128,128])




