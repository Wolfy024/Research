import torch


class LinearTimestep(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearTimestep, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1024)
        self.linear2 = torch.nn.Linear(1024, 2048)
        self.linear3 = torch.nn.Linear(2048, output_size)

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


if __name__ == "__main__":
    # Create a model
    model = LinearTimestep(1, 26000)
    x = model(torch.randn(1, 1))
    print(x.size())