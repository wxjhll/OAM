import numpy as np
import matplotlib.pyplot as plt
import cv2
np.random.seed(43)
from tqdm import tqdm


def cart2pol(x, y):
    xx, yy = np.meshgrid(x, y)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    temp_phi = np.arctan2(yy, xx)
    phi = np.arctan2(yy, xx)
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            if temp_phi[i][j] < 0:
                phi[i][j] = temp_phi[i][j] + 2 * np.pi
            else:
                phi[i][j] = temp_phi[i][j]
    return rho, phi


def get_ping(Cn2):
    C = 2 * np.pi / L
    k = 2 * np.pi / bochang  # 波数
    x = np.linspace(-N / 2, N / 2, N)
    y = np.linspace(-N / 2, N / 2, N)
    X, Y = np.meshgrid(x, y)
    l0 = 0.001
    L0 = 50
    km = 5.92 / l0
    k0 = 2 * np.pi / L0
    kr = np.sqrt((2 * np.pi * X / L) ** 2 + (2 * np.pi * Y / L) ** 2)
    pusai = 2 * np.pi * k ** 2 * 0.033 * Cn2 * dz * np.exp(-(kr / km) ** 2) / (kr ** 2 + k0 ** 2) ** (11 / 6)
    pusai = np.fft.fftshift(pusai)
    ra = np.random.randn(N, N)
    rb = np.random.randn(N, N)
    rr = ra + 1j * rb
    ping = np.sqrt(C) * N ** 2 * np.fft.ifft2(rr * np.sqrt(pusai))
    ping = np.real(ping)
    return ping

def get_H():
    dx = L / N
    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
    fy = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
    [Fx, Fy] = np.meshgrid(fx, fy)
    H = np.exp(1j * k * z * np.sqrt(1 - (bochang * Fx) ** 2 - (bochang * Fy) ** 2))
    return H


L = 0.15  # 相位屏长度
N = 128# 采样点数
dx = L / N  # 像素间距
bochang = 1550e-9  # 波长
z = 400  # 传播距离
dz = 400  # 相位屏间隔
k = 2 * np.pi / bochang  # 波数
w0 = 0.03  # 束腰半径
x = np.linspace(-L / 2, L / 2, N)
y = np.linspace(-L / 2, L / 2, N)
X, Y = np.meshgrid(x, y)
[r, theta] = cart2pol(x, y)
beta = 40 * np.pi / 180

m = 0  # 拓扑核
E1 = (r / w0) ** abs(m) * np.exp(-r ** 2 / w0 ** 2) * np.exp(1j * beta) * np.exp(-1j * m * theta)  # 环形涡旋光
cn2=[]
pmax=0
pmin=0

for i in tqdm(range(30000)):
        Cn2 = 1e-13*((i)%10+1) # 湍流强度
        ping = get_ping(Cn2)  # 湍流相位屏
        E = np.fft.fft2(E1 * np.exp(1j * ping))
        H = get_H()  # 传递函数
        H = np.fft.fftshift(H)
        E = np.fft.ifft2(E * H)
        # 湍流后光束
        I_E = abs(E)
        I_E = I_E / np.max(abs(E))
        I_E = I_E * 255
        I_E = I_E.astype(np.uint8)
        # 原光束
        I1 = abs(E1)
        I1 = I1 / np.max(abs(E1))
        I1 = I1 * 255
        I1 = I1.astype(np.uint8)
        '''
        plt.figure()
        plt.imshow(I_E)
        plt.clim(0,255)
        plt.show()
        '''
        if np.max(ping) > pmax:
            pmax = np.max(ping)
        if np.min(ping) < pmin:
            pmin = np.min(ping)


        ping=(ping+ 5.342202200542821)/(4.903877148493669 +5.342202200542821)
        ping=ping*255
        ping = ping.astype(np.uint8)
        path_at='D:/aDeskfile/OAM/AT/'+str(i)+'.png'
        path_ping='D:/aDeskfile/OAM/ping/'+str(i)+'.png'
        cv2.imwrite(path_at, I_E)
        cv2.imwrite(path_ping, ping)


print(pmax,pmin)






