import numpy as np
from matplotlib import pyplot as plt
np.random.seed(12)
N = 128  # 像素点
L = 0.0047*N
x = np.linspace(-N / 2, N / 2, N) * 2 * np.pi / L
y = np.linspace(-N / 2, N / 2, N) * 2 * np.pi / L
X, Y = np.meshgrid(x, y)
max=0
for _ in range(100):
    ra = np.random.randn(N, N)
    rb = np.random.randn(N, N)
    C = ra + 1j * rb

    Cn2 = 7e-14 # 湍流常数
    dz = 1500  # 传播距离
    lamda = 1.55e-6  # 波长
    l0 = 0.01  # 内外尺度
    L0 = 100

    k = 2 * np.pi / lamda  # 波数
    kl = 3.3 / l0
    k0 = 2 * np.pi / L0
    kr = np.sqrt(X**2 + Y**2)
    fai1 = 2 * np.pi * k**2 * dz
    fai2 = 0.033 * Cn2 * (1 + 1.802 * kr / kl - 0.254 * (kr / kl) ** (7 / 6))
    fai3 = (kr**2 + k0**2) ** (-11 / 6) * np.exp(-(kr**2) / (kl**2))
    fai = fai1 * fai2 * fai3 * (2 * np.pi / L) ** 2
    ping = np.fft.fft2(np.fft.fftshift(C * np.sqrt(fai)))
    ping = np.abs(ping)
    # if np.max(ping)>max:
    #     max=np.max(ping)
    # print(max)

    plt.figure(figsize=(8, 6))
    plt.imshow(ping, cmap='jet')
    plt.clim(0,57.44)
    plt.colorbar()
    plt.show()