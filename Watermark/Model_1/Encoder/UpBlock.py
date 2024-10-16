import torch.nn as nn
import torch.nn.functional as F


class ConvUpBlock(nn.Module):
    def __init__(self, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, padding_mode='reflect'):
        super().__init__(ConvUpBlock, self)
        self.lazy_conv1 = nn.LazyConvTranspose2d(out_channels,
                                                 kernel_size,
                                                 stride,
                                                 padding,
                                                 dilation,
                                                 padding_mode=padding_mode)
        self.lazy_instance_norm = nn.LazyInstanceNorm2d()

    def forward(self, x):
        x = F.relu(self.lazy_instance_norm(self.lazy_conv1(x)))
        return x
    