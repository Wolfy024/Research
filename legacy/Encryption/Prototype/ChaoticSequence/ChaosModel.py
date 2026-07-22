import torch.nn as nn
import torch

class ChaosLSTM(nn.Module):
    def __init__(self, in_channels, sequence_length, keys):
        super(ChaosLSTM, self).__init__()
        self.linear1 = nn.Linear(in_channels, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 2048)
        self.linear4 = nn.Linear(2048, 4096)
        self.linear5 = nn.Linear(4096, 8192)
        self.linear6 = nn.Linear(8192, 8192)
        self.linear7 = nn.Linear(8192, sequence_length * keys)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.linear6(x)
        x = self.linear7(x)
        output = x
        return output


if __name__ == "__main__":
    import torch
    encoder = ChaosLSTM(1, 1024, 3)
    y = encoder(torch.tensor(0.213454).unsqueeze(0))
    y = y.view(1024, 3)
    print(y.shape)
