import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from data.make_dataset import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import *
from model import *
from resnet_se import *
import torch.nn as nn
from torchsummary import summary
from classfiy_dataset import *
import torchvision.models as models
def train_model(batch_size = 32,epochs = 50):
    transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        ])

    train_at, train_lable, val_at, val_lable = split_train_val(imgage_dir='/data/home/Deepin/mutiloam/at/*/*',
                                                               split=0.8)
    train_dataset = MyDataset(input_dir=train_at,
                              ground_dir=train_lable,
                              transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True
                                  , num_workers=4, drop_last=False)
    #验证集
    val_dataset = MyDataset(input_dir=val_at,
                              ground_dir=val_lable,
                              transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False
                                  , num_workers=4, drop_last=False)

    device = torch.device("cuda")

    model=SEResNet18(num_classes=10, reduction=16)
    #model.load_state_dict(torch.load('model.pth'))
    model = model.to(device)
    summary(model, (1,224,224))
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    milest = np.linspace(10, epochs, 10)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milest, gamma=0.5)

    model.train()
    train_corect=[]
    val_corect=[]
    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        print("Epoch{}/{}".format(epoch, epochs))
        print("-" * 10)
        for data in train_dataloader:
            X_train, y_train = data
            X_train, y_train = X_train.cuda(), y_train.cuda()
            outputs = model(X_train)
            _, pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = loss_fn(outputs, y_train)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += torch.sum(pred == y_train.data)
        testing_correct = 0

        for data in val_dataloader:
            X_test, y_test = data
            # 有GPU加下面这行，没有不用加
            X_test, y_test = X_test.cuda(), y_test.cuda()
            outputs = model(X_test)
            _, pred = torch.max(outputs, 1)
            testing_correct += torch.sum(pred == y_test.data)
        train_acc=100 * running_correct / len(train_at)
        val_acc=100 * testing_correct / len( val_at)
        train_corect.append(train_acc.cpu())
        val_corect.append(val_acc.cpu())
        print("Loss is :{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}".format(running_loss / len(train_at),
                                                                                         train_acc,val_acc))
    plt.plot(range(1, len(train_corect) + 1), train_corect, label='Training Acc')
    plt.plot(range(1, len(val_corect) + 1), val_corect, label='Validation Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()
    # plt.show()
    timestamp = int(time.time())
    plt.savefig('./loss_curve{}.png'.format(timestamp))

if __name__ == '__main__':
    train_model(batch_size =32,epochs =50)