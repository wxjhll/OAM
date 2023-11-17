from torchvision import datasets, models, transforms
from data.make_dataset import *
from torch.utils.data import DataLoader
import torch.nn as nn

def get_transformer(size_=(128,128)):
    transform = transforms.Compose([
           transforms.Resize(size_),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])])
    return transform
def get_dateloder(train_dir='',split=0.8,batch_size=16,size_=(128,128)):
    transform=get_transformer(size_)
    train_at, train_ping, val_at, val_ping = \
        split_train_val(imgage_dir=train_dir, split=split)

    train_dataset = \
        MyDataset(input_dir=train_at,ground_dir=train_ping,transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True , num_workers=4, drop_last=False)

    val_dataset = MyDataset(input_dir=val_at,ground_dir=val_ping,transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False
                                , num_workers=4, drop_last=False)

    return train_dataloader,val_dataloader

