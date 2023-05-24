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
def ground_path_map(image_dir=None):
    #D:/Ldata/NOAM/train/AT/5.png
    strlist=image_dir.split('\\')
    ground_true='D:/aDeskfile/OAM/ping/'+strlist[-1]
    return ground_true

def split_train_val(imgage_dir='D:/aDeskfile/OAM/AT',split=0.8):
    AT_dir = get_image_paths(imgage_dir)
    ping_dir=list(map(ground_path_map,AT_dir))
    num=len(AT_dir)
    print('数据集大小：',num)
    index = np.array([int(i) for i in range(num)])# test_data为测试数据
    #index=
    np.random.shuffle(index)  # 打乱索引
    AT_dir=np.asarray(AT_dir)
    ping_dir=np.asarray( ping_dir)
    AT_dir = AT_dir[index]
    ping_dir = ping_dir[index]

    splitpoint=int(num*split)
    print('划分点：',splitpoint)
    train_at=AT_dir[0:splitpoint]
    val_at = AT_dir[splitpoint:]
    train_ping=ping_dir[0:splitpoint]
    val_ping=ping_dir[splitpoint:]
    print('训练集：',train_at[0],train_ping[0])
    print('验证集：', val_at[1000], val_ping[1000])
    return train_at,train_ping,val_at,val_ping


class MyDataset(Dataset):

    def __init__(self, input_dir,ground_dir, transform=None):
        self.input_dir = input_dir
        self.ground_dir = ground_dir
        self.transform = transform
        self.size = len(input_dir)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input=Image.open(self.input_dir[idx])
        #input=np.asarray(input)
        ground_true = Image.open(self.ground_dir[idx])
        #ground_true = np.asarray(ground_true)
        #ground_true = torchvision.transforms.Resize()(ground_true)
        if self.transform:
            input = self.transform(input)
            ground_true = torchvision.transforms.ToTensor()(ground_true)

        return input,ground_true


if __name__ == '__main__':
    #inputdir=get_image_paths('D:/Ldata/NOAM/AT')
    #grounddir=get_image_paths('D:/Ldata/NOAM/ping')
    #print(inputdir)
    train_at, train_ping, val_at, val_ping=split_train_val(imgage_dir='D:/aDeskfile/OAM/AT',split=0.9)
    train_dataset = MyDataset(input_dir=train_at,
                              ground_dir=train_ping,
                              transform=None)

    plt.figure()
    for i,(input,ground) in enumerate(train_dataset):

        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        ax1.axis('off')
        ax1.imshow(input,cmap='jet')
        ax2.axis('off')
        ax2.imshow(ground,cmap='jet')
        plt.show()
        #ax1.set_title('label {}'.format(label))
        #plt.pause(0.001)


