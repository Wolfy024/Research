import torch.nn as nn
import torch.nn.functional as F
from Encryption.Model2.UNET.DownBlock import ConvBlock
from torch import cat

class ConvUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super(ConvUpBlock, self).__init__()
        self.up_conv1 = nn.ConvTranspose2d(in_channels,
                                           out_channels,
                                           kernel_size,
                                           stride,
                                           padding)
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        if y is None:
            return self.relu(self.up_conv1(x))
        x = self.up_conv1(x)
        x = self.conv_block(cat((x, y), 1))
        return x
