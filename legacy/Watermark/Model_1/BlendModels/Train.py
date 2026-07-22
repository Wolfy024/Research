import torch
import matplotlib.pyplot as plt


def imshow(original, output, watermark, title=None):
    """Displays the original image, output image, and watermark side by side."""
    plt.figure(figsize=(12, 4))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original.permute(1, 2, 0).cpu())  # Convert CHW to HWC
    plt.title('Original Image')
    plt.axis('off')

    # Output Image
    plt.subplot(1, 3, 2)
    plt.imshow(output.permute(1, 2, 0).cpu())  # Convert CHW to HWC
    plt.title('Output Image')
    plt.axis('off')

    # Watermark Image
    plt.subplot(1, 3, 3)
    plt.imshow(watermark.permute(1, 2, 0).cpu())  # Convert CHW to HWC
    plt.title('Watermark')
    plt.axis('off')

    if title:
        plt.suptitle(title)
    plt.show()


def train(TRAIN_LOADER, VAL_LOADER, total_epochs, current_epoch, model_encoder, model_decoder, optimizer_encoder,
          optimizer_decoder, criterion):
    import config
    model_encoder.train()
    model_decoder.train()
    running_loss_encoder = 0.0
    running_loss_decoder = 0.0
    num_batches = 0

    for batch, (image, watermark) in enumerate(TRAIN_LOADER):
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()

        og = image.to(config.device)
        watermark = watermark.to(config.device)

        output_encoder = model_encoder(og, watermark)
        output_decoder = model_decoder(output_encoder)

        loss_encoder = criterion(output_encoder, og)
        loss_decoder = criterion(output_decoder, watermark) * 2

        # Combine losses as per your logic
        total_loss_encoder = loss_encoder + loss_decoder

        # Backward pass for the encoder's loss (retain the graph for the decoder's loss)
        total_loss_encoder.backward()

        # Step optimizers
        optimizer_encoder.step()
        optimizer_decoder.step()

        # Update running loss
        running_loss_encoder += loss_encoder.item()
        running_loss_decoder += loss_decoder.item()
        num_batches += 1

        if (batch + 1) % 100 == 0:
            # Perform validation
            validate_encoder(model_encoder, VAL_LOADER, criterion)
            validate_decoder(model_encoder, model_decoder, VAL_LOADER, criterion)

            # Calculate average losses
            average_loss_encoder = running_loss_encoder / num_batches
            average_loss_decoder = running_loss_decoder / num_batches

            print(f"Epoch [{current_epoch + 1}/{total_epochs}], Step [{batch + 1}/{len(TRAIN_LOADER)}], "
                  f"Running Loss Encoder: {average_loss_encoder:.4f}, Running Loss Decoder: {average_loss_decoder:.4f}")

    return running_loss_encoder, running_loss_decoder


def validate_encoder(model_encoder, val_loader, criterion):
    model_encoder.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    num_batches = 0
    import config
    with torch.no_grad():  # No need to compute gradients during validation
        for image, watermark in val_loader:  # Assuming watermarks are not needed for encoder validation
            image = image.to(config.device)
            watermark = watermark.to(config.device)
            output = model_encoder(image, watermark)  # Get the output from the encoder

            # Calculate loss
            loss = criterion(output, image)
            running_loss += loss.item()
            num_batches += 1

            # Show images for the first batch only
            if num_batches == 1:
                imshow(image[0], output[0], watermark[0], title='Encoder Validation')  # Display the first image

    average_loss = running_loss / num_batches if num_batches > 0 else 0
    print(f"Validation Loss (Encoder): {average_loss:.4f}")
    return average_loss


def validate_decoder(model_encoder, model_decoder, val_loader, criterion):
    model_encoder.eval()  # Set the model to evaluation mode
    model_decoder.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    num_batches = 0
    import config
    with torch.no_grad():  # No need to compute gradients during validation
        for image, watermark in val_loader:  # Assuming you have a dataset with watermarks
            image = image.to(config.device)
            watermark = watermark.to(config.device)
            output = model_encoder(image, watermark)  # Get the output from the encoder
            actual_output = model_decoder(output)  # Get the output from the decoder

            # Calculate loss
            loss = criterion(actual_output, watermark)
            running_loss += loss.item()
            num_batches += 1

            # Show images for the first batch only
            if num_batches == 1:
                imshow(output[0], actual_output[0], watermark[0], title='Decoder Validation')  # Display the first image

    average_loss = running_loss / num_batches if num_batches > 0 else 0
    print(f"Validation Loss (Decoder): {average_loss:.4f}")
    return average_loss


def main(folder1, transforms):
    import MAM_decoder, MAM_encoder
    import config
    import DataStream.data as datastream
    from torch.utils.data import DataLoader, random_split
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model_decoder = MAM_decoder.MAM_decoder(3,3).to(config.device)
    model_encoder = MAM_encoder.MAM_encoder().to(config.device)
    optim_decoder = torch.optim.Adam(model_decoder.parameters())
    optim_encoder = torch.optim.Adam(model_encoder.parameters())
    criterion = torch.nn.MSELoss()
    DATA = datastream.Data(folder1, transform)
    total_size = len(DATA)
    print(total_size)
    train_size = int(total_size * config.TRAIN_RATIO)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(DATA, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    for epoch in range(config.NUM_EPOCHS):
        print(epoch)
        running_loss_encoder, running_loss_decoder = train(train_loader, val_loader, config.NUM_EPOCHS, epoch, model_encoder, model_decoder,optim_encoder, optim_decoder, criterion)
        torch.save(model_decoder.state_dict(), f"Epoch_{epoch}_decoder_loss_{running_loss_decoder}.pt")
        torch.save(model_encoder.state_dict(), f"Epoch_{epoch}_encoder_loss_{running_loss_encoder}.pt")


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
