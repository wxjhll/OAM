import numpy as np
from matplotlib import pyplot as plt
np.random.seed(12)
import math
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


N = 200 # 像素点
L = 0.0003*N
x = np.linspace(-N / 2, N / 2, N)
y = np.linspace(-N / 2, N / 2, N)
X, Y = np.meshgrid(x, y)
max=0
for _ in range(100):
    ra = np.random.randn(N, N)
    rb = np.random.randn(N, N)
    C = ra + 1j * rb

    Cn2 = 1e-14 # 湍流常数
    dz = 20  # 传播距离
    lamda = 1.55e-6  # 波长
    l0 = 0.0001  # 内外尺度
    L0 = 50

    k = 2 * np.pi / lamda  # 波数
    kl = 3.3 / l0
    k0 = 2 * np.pi / L0
    kr = np.sqrt((2 * np.pi * X / L)**2 + (2 * np.pi * Y / L)**2)
    fai1 = 2 * np.pi * k**2 * dz
    fai2 = 0.033 * Cn2 * (1 + 1.802 * kr / kl - 0.254 * (kr / kl) ** (7 / 6))
    fai3 = (kr**2 + k0**2) ** (-11 / 6) * np.exp(-(kr**2) / (kl**2))
    fai = fai1 * fai2 * fai3 * (2 * np.pi / L) ** 2
    ping =np.sqrt(2 * np.pi / L)*N**2*np.fft.ifft2(np.fft.fftshift(C * np.sqrt(fai)))
    ping = np.real(ping)
    # if np.max(ping)>max:
    #     max=np.max(ping)
    # print(max)
    plt.figure(figsize=(8, 6))
    plt.imshow(ping, cmap='jet')
    # plt.clim(0,57.44)
    plt.colorbar()
    plt.show()