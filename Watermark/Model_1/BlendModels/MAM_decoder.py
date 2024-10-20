from torch import nn
from torch import cat
import torch.nn.functional as F


class MAM_decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MAM_decoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, output_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return F.tanh(x)


if __name__ == "__main__":
    import torch
    model = MAM_decoder(3, 3).to('cuda')
    print(model(torch.randn(1, 3, 256, 256).to('cuda')).shape)
