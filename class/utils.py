import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
trainpath = './dataset/train/'
valpath = './dataset/val/'

traintransform = transforms.Compose([
    transforms.Resize([64,64]),
    transforms.ToTensor(),  # 将图片数据变为tensor格式
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
])

valtransform = transforms.Compose([
    transforms.Resize([64,64]),
    transforms.ToTensor(),  # 将图片数据变为tensor格式
])

trainData = dsets.ImageFolder(trainpath, transform=traintransform)  # 读取训练集，标签就是train目录下的文件夹的名字，图像保存在格子标签下的文件夹里
valData = dsets.ImageFolder(valpath, transform=valtransform)
batch_size=16
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
valLoader = torch.utils.data.DataLoader(dataset=valData, batch_size=batch_size, shuffle=False)
