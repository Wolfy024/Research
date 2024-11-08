import torch
import config
from loss_fn import EncryptionLoss
from LinearModel import LinearTimestep
from torch.utils.data import DataLoader, TensorDataset


def train(model, optimizer, criterion, TrainLoader):
    for epoch in range(config.epochs):
        loss = 0
        for batch, (data,) in enumerate(TrainLoader):
            data = data.to(config.device)
            seed = torch.randint(0, 100000, (1, 1), dtype=torch.float32, device=config.device)
            optimizer.zero_grad()
            output = model(data, seed)
            loss = criterion(output)
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}")
        torch.save(model.state_dict(), fr"Models\model_{loss}.pth")


def main():
    data = TensorDataset(torch.randn(10000, 1))
    model = LinearTimestep(1, 256 * 256).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    criterion = EncryptionLoss(225).to(config.device)
    TrainLoader = DataLoader(data, batch_size=32, shuffle=True)
    train(model, optimizer, criterion, TrainLoader)



main()
