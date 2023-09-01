import numpy as np
import math
import cv2 as cv
import  matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(1998)
import  scipy.io as sio
#from buchang import cnn_ping
def lg_light(lambda_val, w0, p, z, theta, r, m):
    k = 2 * np.pi / lambda_val
    f = np.pi * w0 ** 2 / lambda_val
    w_z = w0 * np.sqrt(1 + (z / f) ** 2)

    # Define the Laguerre polynomial function
    def Laguerre(p, m, x):
        if p == 0:
            return 1
        elif p == 1:
            return 1 + m - x
        else:
            return ((2 * p - 1 + m - x) * Laguerre(p - 1, m, x) - (p - 1 + m) * Laguerre(p - 2, m, x)) / p

    a0 = np.sqrt(2 / np.pi) * np.sqrt(math.factorial(p) / math.factorial(p + abs(m))) / w_z * \
         (np.sqrt(2) * r / w_z) ** abs(m) * \
         Laguerre(p, abs(m), 2 * r ** 2 / w_z ** 2) * \
         np.exp(-r ** 2 / w_z ** 2)

    phz = (abs(m) + 2 * p + 1) * np.arctan(z / f) - k * (z + r ** 2 * z / (2 * (z ** 2 + f ** 2))) - m * theta

    E0 = a0 * np.exp(1j * phz)
    I0 = np.abs(E0) ** 2

    return I0, E0

def get_ping(Cn2):
    C = 2 * np.pi / L
    k = 2 * np.pi / lambda_val  # 波数
    x = np.linspace(-Nxy / 2, Nxy / 2, Nxy)
    y = np.linspace(-Nxy / 2, Nxy / 2, Nxy)
    X, Y = np.meshgrid(x, y)
    l0 = 0.0001
    L0 = 50
    km = 5.92 / l0
    k0 = 2 * np.pi / L0
    kr = np.sqrt((2 * np.pi * X / L) ** 2 + (2 * np.pi * Y / L) ** 2)
    pusai = 2 * np.pi * k ** 2 * 0.033 * Cn2 * dz * np.exp(-(kr / km) ** 2) / (kr ** 2 + k0 ** 2) ** (11 / 6)
    pusai = np.fft.fftshift(pusai)
    ra = np.random.randn(Nxy, Nxy)
    rb = np.random.randn(Nxy, Nxy)
    rr = ra + 1j * rb
    ping = np.sqrt(C) * Nxy ** 2 * np.fft.ifft2(rr * np.sqrt(pusai))
    ping = np.real(ping)
    return ping
def get_H():
    dx = L / Nxy
    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), Nxy)
    fy = np.linspace(-1 / (2 * dx), 1 / (2 * dx), Nxy)
    [Fx, Fy] = np.meshgrid(fx, fy)
    H = np.exp(1j * k * dz * np.sqrt(1 - (lambda_val  * Fx) ** 2 - (lambda_val  * Fy) ** 2))
    return H

Nxy = 256
lambda_val = 632e-9
k = 2 * math.pi / lambda_val
w = 0.03
p = 0
z = 0


dz=1500
# Coordinate settings
L = 0.2
x = np.linspace(-L/2, L/2, Nxy)
y = np.linspace(-L/2, L/2, Nxy)
X, Y = np.meshgrid(x, y)
theta, r = np.arctan2(Y, X), np.sqrt(X**2 + Y**2)

del_f = 1 / L
dx = L / Nxy
                 # Azimuthal mode index
I0, E0 = lg_light(lambda_val, w, p, z, theta, r, 3)
H = get_H()  # 传递函数
H = np.fft.fftshift(H)

ping=0
E_c=E0
E_cir=0
matrix = np.zeros((256,256))
# 设置圆的中心坐标和半径
center_x = 128
center_y = 128
radius = 128
# 计算每个点到中心的距离
distance_matrix = np.fromfunction(lambda i, j: np.sqrt((i - center_x)**2 + (j - center_y)**2), matrix.shape)
# 根据距离判断是否在圆内，圆内的点数值设置为0，圆外的点数值设置为1
matrix[distance_matrix<= radius] = 1
max=0
min=0
for num in tqdm(range(5)):
    #Cn2=(num%5+5)*1e-14
    Cn2=1e-13

    ping = get_ping(Cn2)  # 湍流相位屏
    #bu=cnn_ping('../at.png')
    #bu=cv.resize(bu,(256,256))
    # if np.max(ping)>max:
    #     max=np.max(ping)
    # if np.min(ping)<min:
    #     min=np.min(ping)
    #ping = (ping % (2 * np.pi))
    # E2 = np.fft.fft2(E_c * np.exp(1j * ping))
    # E = np.fft.ifft2(E2 * H)

    E_cir = np.fft.fft2(E_c * np.exp(1j * ping))
    E_cir = np.fft.ifft2(E_cir * H)

    I=np.abs( E_cir) ** 2
    I=(I/np.max(I)*255).astype(np.uint8)
    #cv.imwrite('../at.png', I)
    #cv.imwrite('D:/aDeskfile/oam_mat/at/{}.png'.format(num), I)
    #8.538194594668175 -7.630559951549822
    # slm_phase=255*(ping+7.630559951549822)/(8.538194594668175+7.630559951549822)
    # slm_phase = (ping/(2 * np.pi)) *255
    # slm_phase = slm_phase.astype(np.uint8)
    ping_mat={'ping':ping}
    #sio.savemat('D:/aDeskfile/oam_mat/ping/{}.mat'.format(num), ping_mat)
    #cv.imwrite('D:/aDeskfile/su_oam/ping/{}.png'.format(num), slm_phase)

# I = np.abs(E) ** 2
# I_cir = np.abs(E_cir) ** 2
#

    # 创建一个2x2的图形
    plt.figure(figsize=(6, 4))
    # 绘制子图1
    plt.subplot(2, 2, 1)
    plt.imshow(ping, cmap='jet')
    plt.title('ping')
    plt.colorbar(aspect=10)
    # 绘制子图2
    plt.subplot(2, 2, 2)
    plt.imshow(I, cmap='jet')
    plt.title('I')
    plt.colorbar(aspect=10)
    # 绘制子图3
    # plt.subplot(2, 2, 3)
    # plt.imshow(data3, cmap='inferno')
    # plt.title('Subplot 3')
    # plt.colorbar(aspect=10)
    # # 绘制子图4
    # plt.subplot(2, 2, 4)
    # plt.imshow(data4, cmap='magma')
    # plt.title('Subplot 4')
    # plt.colorbar(aspect=10)
    # 调整子图之间的间距
    plt.tight_layout()
    # 显示图形
    plt.show()






