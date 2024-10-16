import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, padding_mode='reflect'):
        super().__init__(ConvBlock, self)
        self.lazy_conv1 = nn.LazyConv2d(out_channels,
                                        kernel_size,
                                        stride,
                                        padding,
                                        dilation,
                                        padding_mode=padding_mode)
        self.lazy_conv2 = nn.LazyConv2d(out_channels,
                                        kernel_size,
                                        stride,
                                        padding,
                                        dilation,
                                        padding_mode=padding_mode)
        self.instanceNorm = nn.LazyInstanceNorm2d()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, max_pool=False):
        if max_pool:
            x = self.pool(x)
        x = F.relu(self.instanceNorm(self.lazy_conv1(x)))
        x = F.relu(self.instanceNorm(self.lazy_conv2(x)))
        return x
