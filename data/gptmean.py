import cv2
import numpy as np
import os
DATA_DIR = 'D:/aDeskfile/OAM/AT'
from tqdm import  tqdm
# 1. 加载数据集中所有图片，并计算它们的均值和方差
mean = np.zeros((1,))
std = np.zeros((1,))
count = 0

for root, dirs, files in os.walk(DATA_DIR):
    for f in tqdm(files):
        if f.endswith('.jpg') or f.endswith('.png'):  # 只处理图片文件
            filepath = os.path.join(root, f)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) /255.# 以灰度模式读取图片
            #image=cv2.resize(image,(64,64))

            # 将图片转换为 (h, w) 的形式
            image = np.expand_dims(image, axis=2)

            # 将每个通道的所有像素值相加
            mean += np.mean(image, axis=(0, 1))
            std += np.std(image, axis=(0, 1))

            count += 1

# 2. 计算均值和标准差的平均值
mean = mean / count
std = std / count

# 3. 输出结果
print('Mean: ', mean)
print('Std: ', std)

#Mean:  [0.12622979]
#Std:  [0.20764818]