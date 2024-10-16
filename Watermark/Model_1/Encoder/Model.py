import torch.nn as nn
from torch import cat
import torch.nn.functional as F
from DownBlock import ConvBlock
from UpBlock import ConvUpBlock


class UNET(nn.Module):
    def __init__(self):
        super().__init__(self, UNET)
        self.down1 = ConvBlock.ConvDownBlock(64, max_pool=False)
        self.down2 = ConvBlock.ConvDownBlock(128, max_pool=True)
        self.down3 = ConvBlock.ConvDownBlock(256, max_pool=True)
        self.bottleneck = ConvBlock.ConvDownBlock(512, max_pool=True)
        self.up1 = ConvUpBlock(256)
        self.up2 = ConvUpBlock(128)
        self.up3 = ConvUpBlock(64)
        self.up4 = ConvUpBlock(3)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.bottleneck(x3)
        x = self.up1(cat((x, x3), 1))
        x = self.up2(cat((x, x2), 1))
        x = self.up3(cat((x, x1), 1))
        x = self.up4(cat((x, x), 1))
        return x
