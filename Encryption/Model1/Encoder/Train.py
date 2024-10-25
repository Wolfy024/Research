

def train(encoder, criterion, optimizer, data_loader, num_epochs):
    encoder.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, data in enumerate(data_loader):
            data = data[0].to(torch.float32)  # Ensure input is float
            optimizer.zero_grad()
            output = encoder(data)
            loss = criterion(output)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 10 == 0:  # Print loss every 10 batches
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

        # Print average loss for each epoch
        avg_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        torch.save(encoder.state_dict(), f"Models/encoder_{epoch + 1}_{avg_loss}.pt")


# Main function for testing
if __name__ == "__main__":
    from torch import optim
    from Encryption.Model1.Encoder.Encoder import Encoder
    from Encryption.Model1.Encoder.lossfn import MaxVarianceLossWithPenalty
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Dummy data for training
    data = torch.rand(512, 1)  # 100 samples, 1 feature (change according to your input)
    data_loader = DataLoader(TensorDataset(data), batch_size=8, shuffle=True)
    encoder = Encoder(1, 3, 256, 256)
    criterion = MaxVarianceLossWithPenalty()
    optimizer = optim.Adam(encoder.parameters(), lr=0.0002)
    # Train the model
    train(encoder, criterion, optimizer, data_loader, num_epochs=5)
