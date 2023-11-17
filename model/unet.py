import torch
# from unet_blocks import *
import torch.nn as nn
# from utils import init_weights


class UpBlock(nn.Module):
    def __init__(self, input_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(input_channels, input_channels // 2, kernel_size=1)
        self.res = ResBlock(input_channels // 2, input_channels // 2)
        # self.up.apply(init_weights)
        # self.res.apply(init_weights)

    def forward(self, input, bridge):
        input = self.up(input)
        input = torch.cat([input, bridge], 1)
        input = self.conv(input)
        input = self.res(input)
        return input


class DownBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DownBlock, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        )
        # self.down.apply(init_weights)

    def forward(self, input):
        return self.down(input)


class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        #self.layers.apply(init_weights)
        # self.conv = nn.Conv2d(output_channels, input_channels, kernel_size=1)

    def forward(self, input):
        output = self.layers(input)
        # if output.shape[1] != input.shape[1]:
        #     output = self.conv(output)
        return output + input


class Unet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_classes, num_filters=[48, 96, 192, 384, 768],
                 apply_last_layer=True, padding=True):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]
            # encoder部分
            self.contracting_path.append(DownBlock(input, output))

        # decoder部分
        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 1
        for i in range(n, 0, -1):
            input = self.num_filters[i]
            self.upsampling_path.append(UpBlock(input))

        self.last_layer = nn.ModuleList()
        if self.apply_last_layer:
            self.last_layer.append(nn.ConvTranspose2d(self.num_filters[0], self.num_filters[0], kernel_size=2, stride=2))
            self.last_layer.append(nn.Conv2d(self.num_filters[0], num_classes, kernel_size=1))
            # nn.init.kaiming_normal_(self.last_layer.weight, mode='fan_in',nonlinearity='relu')
            # nn.init.normal_(self.last_layer.bias)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i - 1])

        del blocks

        if self.apply_last_layer:
            x = self.last_layer[0](x)
            x = self.last_layer[1](x)

        return x