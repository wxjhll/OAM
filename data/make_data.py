from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import glob
import pathlib
import numpy as np
import matplotlib.pyplot as plt
def get_image_paths(image_dir=None):
    data_path = pathlib.Path(image_dir)
    paths = list(data_path.glob('*'))
    paths = [str(p) for p in paths]
    #train_data_path = glob.glob( 'E:/data/train/*/*.jpg' )
    return paths
def ground_path_map(image_dir=None):
    #D:/Ldata/NOAM/train/AT/5.png
    strlist=image_dir.split('/')
    ground_true=r'D:/Ldata/NOAM/train/ping'+strlist[-1]
    return ground_true

def split_train_val(imgage_dir=None):
    num=len(imgage_dir)
    return

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
        input=np.asarray(input)/255.
        ground_true = Image.open(self.ground_dir[idx])
        ground_true = np.asarray(ground_true)/255.

        if self.transform:
            input = self.transform(input)

        return input,ground_true


if __name__ == '__main__':
    inputdir=get_image_paths('D:/Ldata/NOAM/train/AT')
    grounddir=get_image_paths('D:/Ldata/NOAM/train/Ping')
    #print(inputdir)

    train_dataset = MyDataset(input_dir=inputdir,
                              ground_dir=grounddir,
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


