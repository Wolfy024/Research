import torch.nn as nn
from torch import cat
import torch.nn.functional as F
from Encryption.Model2.UNET.DownBlock import ConvBlock
from Encryption.Model2.UNET.UpBlock import ConvUpBlock
import torch

class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()  # Corrected line
        self.down1 = ConvBlock(3, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.bottleneck = ConvBlock(256, 512)
        self.up1 = ConvUpBlock(512, 256)
        self.up2 = ConvUpBlock(256, 128)
        self.up3 = ConvUpBlock(128, 64)
        self.conv_final = ConvUpBlock(64, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x=None, bottleneck=False, reverse=False, x1=None, x2=None, x3=None):
        if reverse:
            x = self.up1(x, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            x = self.conv_final(x)
            return F.tanh(x)
        x1 = self.down1(x)
        x2 = self.down2(x1, max_pool=True)
        x3 = self.down3(x2, max_pool=True)
        x = self.bottleneck(x3, max_pool=True)
        if bottleneck:
            return [x.clone().detach() for x in (x, x3, x2, x1)]
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.conv_final(x)
        return F.tanh(x)


if __name__ == "__main__":
    import torch
    model = UNET().to('cuda')
    print(model(torch.randn(1, 3, 256, 256).to('cuda')).shape)
