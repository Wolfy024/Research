import torch


def train(model, criterion, optimizer, TRAIN_LOADER, VAL_LOADER, total_epochs, current_epoch):
    import config
    model.train()
    running_loss = 0.0
    num_batches = 0
    for batch, i in enumerate(TRAIN_LOADER):
        optimizer.zero_grad()
        data = i.to(config.device)
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1
        if (batch + 1) % 10 == 0:
            validate(model, VAL_LOADER)
            average_loss = running_loss / num_batches
            print(f"Epoch [{current_epoch + 1}/{total_epochs}], Step [{batch + 1}/{len(TRAIN_LOADER)}], "
                  f"Running Loss: {average_loss:.4f}")
    return running_loss


def validate(model, val_loader):
    import matplotlib.pyplot as plt
    import config
    import torchvision.utils as vutils
    model.eval()
    with torch.no_grad():
        for batch, inputs in enumerate(val_loader):
            inputs = inputs.to(config.device)
            output = model(inputs)
            # Show the first 5 images from the batch
            if batch == 0:  # Only display images from the first batch
                # Combine input and output images side by side for each example
                combined_images = torch.cat([inputs[:5], output[:5]], dim=3)  # Concatenate along width (dim=3)

                # Make a grid of images with input-output pairs side by side
                grid_combined = vutils.make_grid(combined_images, nrow=5, normalize=True)

                # Plot the combined images
                plt.figure(figsize=(15, 5))  # Adjust figure size for side-by-side comparison
                plt.title("Input and Output Images Side by Side")
                plt.imshow(grid_combined.permute(1, 2, 0).cpu().numpy())
                plt.axis('off')

                plt.show()
                break


def main(folder1, transforms):
    import Model
    import config
    import Watermark.Model_1.Encoder.DataStream.datastream as datastream
    from torch.utils.data import DataLoader, random_split
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = Model.UNET().to(config.device)
    optim = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    DATA = datastream.DataStream(folder1, transform)
    total_size = len(DATA)
    print(total_size)
    train_size = int(total_size * config.TRAIN_RATIO)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(DATA, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    for epoch in range(config.NUM_EPOCHS):
        print(epoch)
        running_loss = train(model, criterion, optim, train_loader, val_loader, config.NUM_EPOCHS, epoch)
        torch.save(model.state_dict(), f"Epoch_{epoch}_UNET_loss_{running_loss}.pt")


if __name__ == "__main__":
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
    ])
    from PreProcessingFunctions import RemoveExtraFiles

    folder1 = r"C:\Users\viraj\PycharmProjects\Research\data"
    RemoveExtraFiles.remove_extra_files(folder1, extensions='.mat')
    main(folder1, transform)
