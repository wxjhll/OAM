import numpy as np
import math
import cv2 as cv
import  matplotlib.pyplot as plt
from tqdm import tqdm
#np.random.seed(1998)

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
    L0 = 2000
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
def Ger_Sax_algo(img,img_at,max_iter):
    h, w = img.shape
    pm_s = np.ones((h, w))
    pm_f = 0
    am_s = np.sqrt(img)
    am_f =np.sqrt(img_at)

    signal_s = am_s*np.exp(pm_s * 1j)

    for iter in range(max_iter):
        signal_f = np.fft.fft2(signal_s)
        pm_f = np.angle(signal_f)
        signal_f =  am_f*np.exp(pm_f * 1j)
        signal_s = np.fft.ifft2(signal_f)
        pm_s = np.angle(signal_s)
        signal_s = am_s*np.exp(pm_s * 1j)

    pm =pm_f
    return pm


Nxy = 256
lambda_val = 632e-9
k = 2 * math.pi / lambda_val
w = 0.06
p = 0
z = 0
m = 3

dz=1000
# Coordinate settings
L = 0.2
x = np.linspace(-L/2, L/2, Nxy)
y = np.linspace(-L/2, L/2, Nxy)
X, Y = np.meshgrid(x, y)
theta, r = np.arctan2(Y, X), np.sqrt(X**2 + Y**2)

del_f = 1 / L
dx = L / Nxy
                 # Azimuthal mode index
m='3-2'
I0, E0 = lg_light(lambda_val, w, p, z, theta, r, 0)
I1, E1 = lg_light(lambda_val, w, p, z, theta, r, -2)
I2, E2 = lg_light(lambda_val, w, p, z, theta, r, 7)
H = get_H()  # 传递函数
H = np.fft.fftshift(H)
Cn2=1e-13
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

for num in tqdm(range(10000)):

    ping = get_ping(Cn2)  # 湍流相位屏
    ping = (ping % (2 * np.pi))

    # E2 = np.fft.fft2(E_c * np.exp(1j * ping))
    # E = np.fft.ifft2(E2 * H)

    ping = ping*matrix
    E_cir = np.fft.fft2(E_c * np.exp(1j * ping))
    E_cir = np.fft.ifft2(E_cir * H)

    I=np.abs( E_cir) ** 2
    I=(I/np.max(I)*255).astype(np.uint8)
    cv.imwrite('D:/aDeskfile/su_oam/at/{}.png'.format(num), I)
    slm_phase = (ping/(2 * np.pi)) *255
    slm_phase = slm_phase.astype(np.uint8)
    cv.imwrite('D:/aDeskfile/su_oam/ping/{}.png'.format(num), slm_phase)

# I = np.abs(E) ** 2
# I_cir = np.abs(E_cir) ** 2
#
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(I, cmap='jet')
# plt.subplot(1, 3, 2)
# plt.imshow(I_cir, cmap='jet')
# plt.subplot(1, 3, 3)
# plt.imshow(matrix, cmap='gray')
# plt.show()





