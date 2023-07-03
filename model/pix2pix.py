import torch.nn as nn
import torch
from collections import OrderedDict


# 定义降采样部分
class downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downsample, self).__init__()
        self.down = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.down(x)


# 定义上采样部分
class upsample(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out=False):
        super(upsample, self).__init__()
        self.up = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5) if drop_out else nn.Identity()
        )

    def forward(self, x):
        return self.up(x)


# ---------------------------------------------------------------------------------
# 定义pix_G  =>input 128*128
class pix2pixG_128(nn.Module):
    def __init__(self):
        super(pix2pixG_128, self).__init__()
        # down sample
        self.down_1 = nn.Conv2d(3, 64, 4, 2, 1)  # [batch,3,128,128]=>[batch,64,64,64]
        for i in range(7):
            if i == 0:
                self.down_2 = downsample(64, 128)  # [batch,64,64,64]=>[batch,128,32,32]
                self.down_3 = downsample(128, 256)  # [batch,128,32,32]=>[batch,256,16,16]
                self.down_4 = downsample(256, 512)  # [batch,256,16,16]=>[batch,512,8,8]
                self.down_5 = downsample(512, 512)  # [batch,512,8,8]=>[batch,512,4,4]
                self.down_6 = downsample(512, 512)  # [batch,512,4,4]=>[batch,512,2,2]
                self.down_7 = downsample(512, 512)  # [batch,512,2,2]=>[batch,512,1,1]

        # up_sample
        for i in range(7):
            if i == 0:
                self.up_1 = upsample(512, 512)  # [batch,512,1,1]=>[batch,512,2,2]
                self.up_2 = upsample(1024, 512, drop_out=True)  # [batch,1024,2,2]=>[batch,512,4,4]
                self.up_3 = upsample(1024, 512, drop_out=True)  # [batch,1024,4,4]=>[batch,512,8,8]
                self.up_4 = upsample(1024, 256)  # [batch,1024,8,8]=>[batch,256,16,16]
                self.up_5 = upsample(512, 128)  # [batch,512,16,16]=>[batch,128,32,32]
                self.up_6 = upsample(256, 64)  # [batch,256,32,32]=>[batch,64,64,64]

        self.last_Conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.init_weight()

    def init_weight(self):
        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_in')
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)

    def forward(self, x):
        # down
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        down_6 = self.down_6(down_5)
        down_7 = self.down_7(down_6)
        # up
        up_1 = self.up_1(down_7)
        up_2 = self.up_2(torch.cat([up_1, down_6], dim=1))
        up_3 = self.up_3(torch.cat([up_2, down_5], dim=1))
        up_4 = self.up_4(torch.cat([up_3, down_4], dim=1))
        up_5 = self.up_5(torch.cat([up_4, down_3], dim=1))
        up_6 = self.up_6(torch.cat([up_5, down_2], dim=1))
        out = self.last_Conv(torch.cat([up_6, down_1], dim=1))
        return out


# 定义pix_D_128    => input 128*128
class pix2pixD_128(nn.Module):
    def __init__(self):
        super(pix2pixD_128, self).__init__()

        # 定义基本的卷积\bn\relu
        def base_Conv_bn_lkrl(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )

        D_dic = OrderedDict()
        in_channels = 6
        out_channels = 64
        for i in range(4):
            if i < 3:
                D_dic.update({'layer_{}'.format(i + 1): base_Conv_bn_lkrl(in_channels, out_channels, 2)})
            else:
                D_dic.update({'layer_{}'.format(i + 1): base_Conv_bn_lkrl(in_channels, out_channels, 1)})
            in_channels = out_channels
            out_channels *= 2
        D_dic.update({'last_layer': nn.Conv2d(512, 1, 4, 1, 1)})  # [batch,1,14,14]
        self.D_model = nn.Sequential(D_dic)

    def forward(self, x1, x2):
        in_x = torch.cat([x1, x2], dim=1)
        return self.D_model(in_x)


# ---------------------------------------------------------------------------------

