import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from data.make_data import *
from model import unet1
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.net import UNet
import torch.nn as nn



def getdataloader(batch_size=32,transform=None):
    inputdir = get_image_paths('D:/Ldata/NOAM/train/AT')
    grounddir = get_image_paths('D:/Ldata/NOAM/train/Ping')
    train_dataset = MyDataset(input_dir=inputdir,
                              ground_dir=grounddir,
                              transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    return train_dataloader
def train_model(batch_size = 32,epochs = 100):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    train_dataloader=getdataloader(batch_size=batch_size,transform=transform)

    learning_rate = 1e-4
    device = torch.device("cuda")

    model=UNet()
    model = model.to(device)
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':
    train_model()
