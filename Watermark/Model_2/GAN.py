from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1024, 512, 3, 1, 1)
        self.conv3 = nn.Conv2d(512, 256, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 3, 3, 1, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.bn1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu2(self.bn2(x))
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
