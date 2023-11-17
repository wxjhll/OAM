import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import torchvision
warnings.filterwarnings(action='ignore')
from fightingcv_attention.attention.CBAM import CBAMBlock
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, reduction=16):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.GroupNorm(32, out_ch),

            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.GroupNorm(32, out_ch),

            nn.ReLU()
        )
        self.se = SELayer(out_ch, reduction)
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            # nn.GroupNorm(32, out_ch),
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.se(x)

        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x =x+residual
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self,in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2,in_ch//2,2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False, rate=0.1):
        super(outconv, self).__init__()
        self.dropout = dropout
        if dropout:
            print('dropout', rate)
            self.dp = nn.Dropout2d(rate)
        self.conv1 = nn.Conv2d(in_ch, in_ch//2, 1)
        self.conv2 = nn.Conv2d(in_ch//2, out_ch, 1)

    def forward(self, x):
        if self.dropout:
            x = self.dp(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SeResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False, dropout=False, rate=0.1):
        super(SeResUNet, self).__init__()
        self.deep_supervision = deep_supervision
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(512 + 512, 256)
        self.up2 = up(256 + 256, 128)
        self.up3 = up(128 + 128, 64)
        self.up4 = up(64 + 64, 64)
        self.outc = outconv(64, n_classes, dropout, rate)
        self.dsoutc4 = outconv(256, n_classes)
        self.dsoutc3 = outconv(128, n_classes)
        self.dsoutc2 = outconv(64, n_classes)
        self.dsoutc1 = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x44 = self.up1(x5, x4)
        x33 = self.up2(x44, x3)
        x22 = self.up3(x33, x2)
        x11 = self.up4(x22, x1)
        x0 = self.outc(x11)
        x0 = nn.ReLU()(x0)
        # if self.deep_supervision and self.training:
        #     x11 = F.interpolate(self.dsoutc1(x11), x0.shape[2:], mode='bilinear')
        #     x22 = F.interpolate(self.dsoutc2(x22), x0.shape[2:], mode='bilinear')
        #     x33 = F.interpolate(self.dsoutc3(x33), x0.shape[2:], mode='bilinear')
        #     x44 = F.interpolate(self.dsoutc4(x44), x0.shape[2:], mode='bilinear')
        #
        #     return x0, x11, x22, x33, x44
        # else:
        return x0


class Shallow_SeResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False, dropout=False, rate=0.1):
        super(Shallow_SeResUNet, self).__init__()
        self.deep_supervision = deep_supervision
        #big
        # self.inc = inconv(n_channels, 64)
        # self.down1 = down(64, 64)
        # self.down2 = down(64, 128)
        # self.down3 = down(128, 256)
        # self.down4 = down(256, 256)
        # self.up1 = up(256 + 256, 128)
        # self.up2 = up(128 + 128, 128)
        # self.up3 = up(128 + 64, 64)
        # self.up4 = up(64 + 64, 64)
        # self.outc = outconv(64, n_classes, dropout, rate)
        # self.dsoutc4 = outconv(128, n_classes)
        # self.dsoutc3 = outconv(128, n_classes)
        # self.dsoutc2 = outconv(64, n_classes)
        # self.dsoutc1 = outconv(64, n_classes)

        # self.down_pre1= down(32, 32)
        # self.down_pre2 = down(32, 32)
        # self.up_after1=nn.ConvTranspose2d(32, 32, 2, stride=2)
        # self.up_after2=nn.ConvTranspose2d(32, 32, 2, stride=2)
        #
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 128)
        self.up1 = up(128 + 128, 64,bilinear=True)
        self.up2 = up(64 + 64, 64,bilinear=True)
        self.up3 = up(64 + 32, 32,bilinear=True)
        self.up4 = up(32 + 32, 32,bilinear=True)

        self.outc = outconv(32, n_classes, dropout, rate)
        # self.dsoutc4 = outconv(64, n_classes)
        # self.dsoutc3 = outconv(64, n_classes)
        # self.dsoutc2 = outconv(32, n_classes)
        # self.dsoutc1 = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # x_pre1=self.down_pre1(x1)
        # x_pre2 = self.down_pre2(x_pre1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x44 = self.up1(x5, x4)
        x33 = self.up2(x44, x3)
        x22 = self.up3(x33, x2)
        x11 = self.up4(x22, x1)

        # after1=self.up_after1(x11)
        # after2 = self.up_after2(after1)
        x0 = self.outc(x11)
        x0 = nn.ReLU()(x0)
        # if self.deep_supervision and self.training:
        #     x11 = F.interpolate(self.dsoutc1(x11), x0.shape[2:], mode='bilinear')
        #     x22 = F.interpolate(self.dsoutc2(x22), x0.shape[2:], mode='bilinear')
        #     x33 = F.interpolate(self.dsoutc3(x33), x0.shape[2:], mode='bilinear')
        #     x44 = F.interpolate(self.dsoutc4(x44), x0.shape[2:], mode='bilinear')
        #
        #     return x0, x11, x22, x33, x44
        # else:
        return x0


if __name__ == '__main__':
    # import os
    # def weights_init(m):
    #     classname = m.__class__.__name__
    #     # print(classname)
    #     if classname.find('Conv') != -1:
    #         torch.nn.init.xavier_uniform_(m.weight.data)
    #         if m.bias is not None:
    #             torch.nn.init.constant_(m.bias.data, 0.0)

    # model = SeResUNet(3, 3, deep_supervision=True).cuda()
    model = Shallow_SeResUNet(n_channels=1, n_classes=1, deep_supervision = False,
                            dropout = False, rate = 0.3)
    # model.apply(weights_init)
    x = torch.randn((1, 1, 128, 128)).cuda()


    # for i in range(1000):
    #     y0, y1, y2, y3, y4 = model(x)
    #     print(y0.shape, y1.shape, y2.shape, y3.shape, y4.shape)

# print(model.weight)

# for name, param in model.named_parameters():
#     print(name, param.requires_grad)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# loss_func = torch.nn.CrossEntropyLoss()
#
# # model.apply(weights_init)
# # # model.load_state_dict(torch.load('./MODEL.pth'))
# model = model.cuda()
# print('load done')
# input()
#
