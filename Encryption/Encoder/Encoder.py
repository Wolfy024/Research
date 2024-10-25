import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, IMG_WIDTH, IMG_HEIGHT):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, 512 * out_channels)
        self.linear2 = nn.Linear(512 * out_channels, 1024 * out_channels)
        self.linear3 = nn.Linear(1024 * out_channels, IMG_WIDTH * IMG_HEIGHT * out_channels)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        output = torch.round(torch.abs(torch.tanh(x) * 1022))
        return output


if __name__ == "__main__":
    import torch

    encoder = Encoder(1, 3, 256, 256)
    y = encoder(torch.tensor(0.213454).unsqueeze(0))
    y = y.view(3, 256, 256)
    y = y.detach().cpu().numpy()
    print(y)
