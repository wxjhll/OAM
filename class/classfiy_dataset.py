import torchvision.transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import glob
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
seed = 1024
random.seed(seed)     # python的随机性
np.random.seed(seed)  # np的随机性
os.environ['PYTHONHASHSEED'] = str(seed) # 设置python哈希种子，为了禁止hash随机化

def get_image_paths(image_dir=None):
    data_path = pathlib.Path(image_dir)
    paths = list(data_path.glob('*'))
    paths = [str(p) for p in paths]
    #train_data_path = glob.glob( 'E:/data/train/*/*.jpg' )
    return paths

def get_class_nums(imgage_dir='/data/home/Deepin/mutiloam/at/*'):
    class_nums=glob.glob(imgage_dir)
    pure_train_labels = set([p.split('/')[-1] for p in class_nums])
    #print(pure_train_labels)
    return pure_train_labels

def split_train_val(imgage_dir='/data/home/Deepin/mutiloam/at/*/*',split=0.8):
    class_nums = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8','L9','L10']
    train_data_path = glob.glob(imgage_dir)
    random.shuffle(train_data_path)
    train_data_lable = [p.split('/')[-2] for p in train_data_path]
    #print(train_data_lable[0:4])

    labels_to_index = dict((index, name) for (name, index) in enumerate(class_nums))
    print(labels_to_index)
    train_labels = [labels_to_index.get(label) for label in train_data_lable]
    #print(train_labels[0:4])
    labels= np.asarray(train_labels)
    #labels = torch.from_numpy(labels)
    #one_hot = torch.nn.functional.one_hot(labels_onehot, num_classes=8)
    num=len(train_data_path)
    index = np.array([int(i) for i in range(num)])  # test_data为测试数据
    # index=
    np.random.shuffle(index)  # 打乱索引
    train_data_path = np.asarray(train_data_path)
    one_hot = labels
    train_data_path = train_data_path[index]
    one_hot = one_hot[index]

    splitpoint = int(num * split)
    print('划分点：', splitpoint)
    train_at = train_data_path[0:splitpoint]
    val_at = train_data_path[splitpoint:]
    train_lable = one_hot[0:splitpoint]
    val_lable = one_hot[splitpoint:]
    print('训练集：', train_at[0], train_lable[0])
    print('验证集：', val_at[10], val_lable[10])

    return train_at,train_lable,val_at,val_lable

class MyDataset(Dataset):

    def __init__(self, input_dir,ground_dir, transform=None):
        self.input_dir = input_dir
        self.lable =torch.from_numpy(ground_dir)
        self.transform = transform
        self.size = len(input_dir)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input=Image.open(self.input_dir[idx])
        #input=np.asarray(input)
        #ground_true = Image.open(self.ground_dir[idx])
        #ground_true = np.asarray(ground_true)
        #ground_true = torchvision.transforms.Resize()(ground_true)
        labels = self.lable[idx]
        #print(labels)
        #torch.from_numpy(self.lable[idx])
        #one_hot_lable = torch.nn.functional.one_hot(labels_onehot, num_classes=8)
        if self.transform:
            input = self.transform(input)

        return input,labels




if __name__ == '__main__':
    class_nums = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8','L9','L10']
    train_at,train_lable,val_at,val_lable=split_train_val(imgage_dir='/data/home/Deepin/mutiloam/at/*/*',
                                                           split=0.8)
    train_dataset = MyDataset(input_dir=train_at,
                              ground_dir=train_lable,
                              transform=None)

    plt.figure()
    for i,(input,ground) in enumerate(train_dataset):

        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        ax1.axis('off')
        ax1.imshow(input,cmap='jet')
        ax2.axis('off')
        ax2.imshow(input,cmap='jet')
        plt.show()
        #ax1.set_title('label {}'.format(label))
        #plt.pause(0.001)


