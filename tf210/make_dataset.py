import pathlib
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(43)
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
def process_image(fpath1, fpath2):
    """ 图片预处理 """#Mean:  [0.12622979]
#Std:  [0.20764818]
    image = tf.io.read_file(fpath1)                  # 读取图像
    image = tf.image.decode_png(image,channels=1)  # jpg图像解码
    image = tf.image.resize(image, [64,64]) /255.0
    image=(image-0.12622979)/0.20764818
    print('image1:',image.shape)
    #image1 = tf.io.read_file(fpath1[1])  # 读取图像
    #image1 = tf.image.decode_jpeg(image1, channels=1)  # jpg图像解码
    #image1=image1/255.0
    #image1 = tf.image.resize(image1, [64,64]) / 255.0  # 原始图片大重设为(x, x), AlexNet的输入是224X224
    #sample=tf.concat([image,image1],axis=-1)

    ground = tf.io.read_file(fpath2)
    ground = tf.image.decode_png(ground, channels=1)
    #ground=ground/255.0
    ground = tf.image.resize(ground, [64,64])/255.0
    #label = tf.one_hot(label, depth=2)
    return image,ground

def get_dataset(image_dir=None, image_dir2=None ,is_shuffle=False, batch_size=1):

    # tensorflow接口创建数据集读取
    ds = tf.data.Dataset.from_tensor_slices((image_dir, image_dir2))
    # 回调数据处理
    ds = ds.map(process_image,num_parallel_calls=4)
    # 洗牌
    if is_shuffle:
         ds = ds.shuffle(buffer_size=2048)
    # 分批次
    ds = ds.batch(batch_size)
    #ds = ds.prefetch(buffer_size=1)
    return ds

def show():
    train_at, train_ping, val_at, val_ping=split_train_val(imgage_dir='D:/aDeskfile/OAM/AT',split=0.8)
    ds = get_dataset(image_dir=train_at, image_dir2=train_ping, is_shuffle=False, batch_size=1)
    for x, y in ds:
        print(x.shape)
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(x[0])
        plt.title('L')
        plt.subplot(1, 3, 2)
        plt.imshow(x[0])
        plt.title('at')
        plt.subplot(1, 3, 3)
        plt.imshow(y[0])
        plt.title('ping')
        plt.show()
        print(y.shape)
if __name__ == '__main__':
    show()
