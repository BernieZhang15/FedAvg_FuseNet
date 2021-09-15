from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter


def conv(in_channels, out_channels, kernel_size=3, padding=1, batch_norm=True):
    c = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
    if batch_norm:
        bn = nn.BatchNorm2d(out_channels)
        return nn.Sequential(c, bn)
    return c


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.conv_in = conv(self.in_channels, self.out_channels)
        self.conv_out = conv(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, integrate=None):
        x = F.relu(self.conv_in(x))
        x = F.relu(self.conv_out(x))
        if integrate is not None:
            x = torch.add(x, integrate)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upconv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=2, stride=2)

        self.conv_in = conv(2 * self.out_channels, self.out_channels)
        self.conv_out = conv(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up, output_size=from_down.size())
        x = torch.cat((from_up, from_down), 1)
        x = F.relu(self.conv_in(x))
        x = F.relu(self.conv_out(x))
        return x


class SegmentationUNet(nn.Module):
    def __init__(self, num_classes, in_channels_dsm=1, in_channels_rgb=4, depth=5, start_filts=64):
        super(SegmentationUNet, self).__init__()

        self.num_classes = num_classes
        self.in_channels_dsm = in_channels_dsm
        self.in_channels_rgb = in_channels_rgb
        self.start_filts = start_filts
        self.depth = depth
        self.down_convs_dsm = []
        self.down_convs_rgb = []
        self.up_convs = []

        outs = 0
        for i in range(depth):
            ins = self.in_channels_dsm if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False
            down_dsm = DownConv(ins, outs, pooling=pooling)
            self.down_convs_dsm.append(down_dsm)

        outs = 0
        for i in range(depth):
            ins = self.in_channels_rgb if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False
            down_rgb = DownConv(ins, outs, pooling=pooling)
            self.down_convs_rgb.append(down_rgb)

        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs)
            self.up_convs.append(up_conv)

        self.down_convs_dsm = nn.ModuleList(self.down_convs_dsm)
        self.down_convs_rgb = nn.ModuleList(self.down_convs_rgb)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.conv_final = conv(outs, self.num_classes, kernel_size=1, padding=0, batch_norm=False)

    def forward(self, dsm, rgb):
        encoder_outs = []
        for module_dsm, module_rgb in zip(self.down_convs_dsm, self.down_convs_rgb):
            dsm, before_pool_dsm = module_dsm(dsm)
            rgb, before_pool_rgb = module_rgb(rgb, before_pool_dsm)
            encoder_outs.append(before_pool_rgb)

        for i, module_up in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            rgb = module_up(before_pool, rgb)
        x = self.conv_final(rgb)
        return x

if __name__ == "__main__":
    x = torch.rand(1, 1, 256, 256)
    y = torch.rand(1, 4, 256, 256)
    model = SegmentationUNet(6)
    for param in model.down_convs_rgb[0].parameters():
        print(param)