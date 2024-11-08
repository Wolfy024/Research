import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from Encryption.Prototype.Compilation.Generate_Sequences import generate_keys, generate_chaotic_sequences


def encrypt_channels(encoder1, encoder2, encoder3, chaos1, chaos2, chaos3, img_pth):
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image_path = img_pth
    image = Image.open(image_path)
    transformed_image = transformations(image).to('cuda')
    time_steps_channel1 = encoder1[0]
    time_steps_channel2 = encoder2[1]
    time_steps_channel3 = encoder3[2]
    output_1 = chaos1.view(-1, 1024, 3)
    output_2 = chaos2.view(-1, 1024, 3)
    output_3 = chaos3.view(-1, 1024, 3)

    # Process each channel
    for channel in range(3):
        counter1, counter2 = 0, 0
        for i in (time_steps_channel1, time_steps_channel2, time_steps_channel3)[channel]:
            for j in i:
                j = j.int()
                new_val = output_1[0][j][channel] if channel == 0 else (
                    output_2[0][j][channel - 1] if channel == 1 else output_3[0][j][channel - 2])

                # Avoid division by zero
                if new_val != 0:
                    transformed_image[channel][counter1][counter2] = (transformed_image[channel][counter1][counter2] +
                                                                      output_1[0][j][(channel + 1) % 3] -
                                                                      output_1[0][j][(channel + 2) % 3]) / new_val
                counter2 += 1
                if counter2 == 255:
                    counter2 = 0
                    break
            counter1 += 1
            if counter1 == 255:
                break
        print(f'Channel {channel + 1} Done.')

    torch.cuda.empty_cache()
    return transformed_image


if __name__ == "__main__":
    transformed_image_cpu = encrypt_channels(0.91354, 0.38435, 5.134545, 4.4565).cpu()
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)  # Mean for each channel
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    denormalized_image = transformed_image_cpu * std + mean
    denormalized_image = torch.clamp(denormalized_image, 0, 1)
    to_pil = transforms.ToPILImage()
    image_from_tensor = to_pil(denormalized_image)
    image_from_tensor.show()
    print(image_from_tensor.size)