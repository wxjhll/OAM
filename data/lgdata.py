import numpy as np
import math
import cv2 as cv
import  matplotlib.pyplot as plt
from tqdm import tqdm
#np.random.seed(1998)
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
    ping = C* Nxy ** 2 * np.fft.ifft2(rr * np.sqrt(pusai))
    ping = np.abs(ping)
    return ping

def get_ping2(Cn2,dz):
    #Hill谱
    N=200
    L=0.0003*N
    lamda = 632e-9  # 波长
    l0 = 0.0001  # 内外尺度
    L0 = 50
    x = np.linspace(-N / 2, N / 2, N)
    y = np.linspace(-N / 2, N / 2, N)
    X, Y = np.meshgrid(x, y)
    for _ in range(1):
        ra = np.random.randn(N, N)
        rb = np.random.randn(N, N)
        C = ra + 1j * rb
        k = 2 * np.pi / lamda  # 波数
        kl = 3.3 / l0
        k0 = 2 * np.pi / L0
        kr = np.sqrt((2 * np.pi * X / L) ** 2 + (2 * np.pi * Y / L) ** 2)
        fai1 = 2 * np.pi * k ** 2 * dz
        fai2 = 0.033 * Cn2 * (1 + 1.802 * kr / kl - 0.254 * (kr / kl) ** (7 / 6))
        fai3 = (kr ** 2 + k0 ** 2) ** (-11 / 6) * np.exp(-(kr ** 2) / (kl ** 2))
        fai = fai1 * fai2 * fai3 * (2 * np.pi / L) ** 2
        ping = np.sqrt(2 * np.pi / L) * N ** 2 * np.fft.ifft2(np.fft.fftshift(C * np.sqrt(fai)))
        ping = np.real(ping)
        return ping
#传递函数
def get_H():
    dx = L / Nxy
    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), Nxy)
    fy = np.linspace(-1 / (2 * dx), 1 / (2 * dx), Nxy)
    [Fx, Fy] = np.meshgrid(fx, fy)
    H = np.exp(1j * k * dz * np.sqrt(1 - (lambda_val  * Fx) ** 2 - (lambda_val  * Fy) ** 2))
    return H
#dx >sqrt(lambda*d/N)
Nxy = 200
lambda_val = 632e-9     #波长
k = 2 * math.pi / lambda_val
w = 0.001   #束腰半径
p = 0   #径向指数
z = 0

dz=40   #传播距离
# Coordinate settings
L =Nxy*0.0003
print(L/Nxy>=np.sqrt(lambda_val*dz/Nxy))

x = np.linspace(-L/2, L/2, Nxy)
y = np.linspace(-L/2, L/2, Nxy)
X, Y = np.meshgrid(x, y)
theta, r = np.arctan2(Y, X), np.sqrt(X**2 + Y**2)
del_f = 1 / L
dx = L / Nxy
beta = 40 * np.pi / 180
m=10
E_g = (r / w) ** abs(m) * np.exp(-r ** 2 / w ** 2) * np.exp(1j * beta) * np.exp(-1j * m * theta)
I0, E0 = lg_light(lambda_val, w, p, z, theta, r, 3)#拉盖尔光束
I_,E_=lg_light(lambda_val, w, p, z, theta, r, -5)
angle_=np.angle(E_+E0)

plt.subplot()
plt.imshow(angle_, cmap='gray')
plt.show()
H = get_H()  # 传递函数
H = np.fft.fftshift(H)
ping=0
E_c=E0
E_cir=0

max=0
min=0
for num in tqdm(range(1)):
    Cn2=1e-14#(num%10+1)*1e-14
    ping = get_ping2(Cn2,dz)  #产生湍流相位屏
    if np.max(ping)>max:
        max=np.max(ping)
    if np.min(ping)<min:
        min=np.min(ping)
    print(max)
    #print(np.var(ping))
    #ping = (ping % (2 * np.pi))
    # E2 = np.fft.fft2(E_c * np.exp(1j * ping))
    # E = np.fft.ifft2(E2 * H)

    E_cir = np.fft.fft2(E_c * np.exp(1j * ping))#大气湍流中传播
    E_cir = np.fft.ifft2(E_cir * H)

    I=np.abs(E_cir) ** 2
    I=(I/np.max(I)*255).astype(np.uint8)
    #cv.imwrite('../at.png', I)
    #cv.imwrite('D:/aDeskfile/OAM/at/{}.png'.format(num), I)
    slm_phase=255*ping/12.074454369318497
    #slm_phase = (ping/(2 * np.pi)) *255
    slm_phase = slm_phase.astype(np.uint8)
    # ping_mat={'ping':ping}
    #sio.savemat('D:/aDeskfile/oam_mat/ping/{}.mat'.format(num), ping_mat)
    #cv.imwrite('D:/aDeskfile/OAM/ping/{}.png'.format(num), slm_phase)

    plt.figure(figsize=(6, 4))
    # 绘制子图1
    plt.subplot()
    plt.imshow(ping, cmap='jet')
    #plt.title('ping')
    #plt.colorbar(aspect=10)
    plt.axis('off')
    plt.savefig('C:/Users/dlmz/Desktop/figure/at.png',bbox_inches='tight'
                ,pad_inches=0.0)
    # 绘制子图2·
    plt.subplot()
    plt.imshow(I, cmap='hot')
    # plt.title('I')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('C:/Users/dlmz/Desktop/figure/I.png', bbox_inches='tight'
                , pad_inches=0.0)
    #绘制子图3
    plt.subplot()
    plt.imshow(I, cmap='gray')
    # plt.title('Subplot 3')
    # plt.colorbar(aspect=10)
    plt.axis('off')
    plt.savefig('C:/Users/dlmz/Desktop/figure/atphase.png', bbox_inches='tight'
                , pad_inches=0.0)
    # 绘制子图4
    plt.subplot()
    plt.imshow(I, cmap='gray')
    # plt.title('Subplot 4')
    # plt.colorbar(aspect=10)
    plt.axis('off')
    plt.savefig('C:/Users/dlmz/Desktop/figure/xiangwei.png', bbox_inches='tight'
                , pad_inches=0.0)
    #调整子图之间的间距
    #plt.tight_layout()
    # 显示图形
    plt.show()
    # fig = plt.figure(figsize=(6, 4))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 生成二维数据
    # data = ping
    #
    # # 创建X和Y坐标轴
    # x = np.arange(0, data.shape[1])
    # y = np.arange(0, data.shape[0])
    # X, Y = np.meshgrid(x, y)
    #
    # # 绘制三维图
    # ax.plot_surface(X, Y, data, cmap='jet')
    #
    # # 添加标签
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    #
    # # 显示图形
    # plt.show()






