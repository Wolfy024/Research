from Encryption.Prototype.data import Datastream, ChaoticData
import random
from ChaosModel import ChaosLSTM
from torch.utils.data import DataLoader
import torch
import config
import numpy as np


def train(model, criterion, optim, DATA_LOADER, num_epochs):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0  # Initialize running loss for this epoch
        for batch, data in enumerate(DATA_LOADER):
            key = data[0].to(config.DEVICE)
            sequence = data[1].to(config.DEVICE)

            optim.zero_grad()  # Zero the gradients
            output = model(key)  # Forward pass

            # Reshape output if necessary
            output = output.view(-1, 1, 1024, 3)  # Adjust based on your model's output shape

            loss = criterion(output, sequence)  # Compute the loss
            loss.backward()  # Backpropagate the loss
            optim.step()  # Update weights

            running_loss += loss.item()  # Accumulate the loss

            # Optional: Print loss every few batches
            if batch % 10 == 0:  # Adjust the frequency as needed
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch}], Loss: {loss.item():.4f}")

        # Print average loss for this epoch
        epoch_loss = running_loss / len(DATA_LOADER)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")
        torch.save(model.state_dict(), fr"Models\model_{epoch + 1}_{epoch_loss}.pt")


def main():
    from lossfn import CustomVarianceLossWithPenalty
    keys = []
    keys = np.array([random.uniform(-20, 20) for _ in range(512)], dtype=np.float32)
    data = ChaoticData.generate_lorenz_data(keys, 1024, initial_range=(-20, 20))
    data_tensor = Datastream.Datastream(keys, data)
    DATA_LOADER = DataLoader(data_tensor, batch_size=config.BATCH_SIZE, shuffle=True)
    model = ChaosLSTM(1, 1024, 3).to(config.DEVICE)
    criterion = CustomVarianceLossWithPenalty()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    train(model, criterion, optimizer, DATA_LOADER, config.EPOCHS)


main()
