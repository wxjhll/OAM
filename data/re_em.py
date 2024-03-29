import numpy as np
import matplotlib.pyplot as plt
import cv2
np.random.seed(42)
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
    l0 = 0.0001
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



N = 256# 采样点数
L = 0.5# 相位屏长度
dx = L / N  # 像素间距
bochang = 632e-9  # 波长
z = 1000 # 传播距离
dz = 200# 相位屏间隔
k = 2 * np.pi / bochang  # 波数
w0 = 0.02 # 束腰半径
k2=k**2
x = np.linspace(-L / 2, L / 2, N)
y = np.linspace(-L / 2, L / 2, N)
X, Y = np.meshgrid(x, y)
[r, theta] = cart2pol(x, y)
beta = 40 * np.pi / 180
Cn2 =1e-13# 湍流强度
r0=(0.423*k2*Cn2*z)**(-3/5)
print('D/r0:',2*w0/r0)
m =3  # 拓扑核
E1 = (r / w0) ** abs(m) * np.exp(-r ** 2 / w0 ** 2) * np.exp(1j * beta) * np.exp(-1j * m * theta)  # 环形涡旋光
E_c=E1
H = get_H()  # 传递函数
H = np.fft.fftshift(H)
for _ in tqdm(range(z//dz)):

    ping = get_ping(Cn2)  # 湍流相位屏
    ping = ping % (2 * np.pi)
    E2 = np.fft.fft2(E_c * np.exp(1j * ping))
    E = np.fft.ifft2(E2 * H)
    E_c=E





    # 湍流后光束
I_E = abs(E_c)
I_E = I_E / np.max(abs(E_c))
I_E = I_E * 255
I_E = I_E.astype(np.uint8)
    # 原光束
I1 = abs(E1)
I1 = I1 / np.max(abs(E1))
I1 = I1 * 255
I1 = I1.astype(np.uint8)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(I_E, cmap='jet')
plt.subplot(1, 3, 2)
plt.imshow(I1, cmap='jet')
plt.subplot(1, 3, 3)
plt.imshow(np.angle(E_c), cmap='gray')
plt.show()











