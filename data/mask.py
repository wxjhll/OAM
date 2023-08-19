import numpy as np
import glob
import cv2 as cv
from tqdm import tqdm

# matrix = np.zeros((512,512))
#     # 设置圆的中心坐标和半径
# center_x = 256
# center_y = 256
# radius = 200
#
# # 计算每个点到中心的距离
# distance_matrix = np.fromfunction(lambda i, j: np.sqrt((i - center_x)**2 + (j - center_y)**2), matrix.shape)
#
# # 根据距离判断是否在圆内，圆内的点数值设置为0，圆外的点数值设置为1
# matrix[distance_matrix<= radius] = 1


train_data_path = glob.glob('D:/aDeskfile/slm/ping_pre/*.png' )
for path in tqdm(train_data_path):
    ping=cv.imread(path,0)
    mask_ping = ping[75:435,75:435]

    # slm_phase = mask_ping.astype(np.uint8)
    path1=path.replace('ping_pre','ping')
    cv.imwrite(path1,mask_ping)
