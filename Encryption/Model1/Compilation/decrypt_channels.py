import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from Generate_Sequences import generate_keys, generate_chaotic_sequences


def decrypt_channels(encoder1, chaos1, chaos2, chaos3, encrypted_image):
    # Initialize the output image tensor
    output_image = encrypted_image.clone()
    time_steps_channel1, time_steps_channel2, time_steps_channel3 = encoder1

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

                # Reverse operation
                if new_val != 0:
                    output_image[channel][counter1][counter2] = new_val * encrypted_image[channel][counter1][counter2] - \
                                                                output_1[0][j][(channel + 1) % 3] + \
                                                                output_1[0][j][(channel + 2) % 3]
                counter2 += 1
                if counter2 == 255:
                    counter2 = 0
                    break
            counter1 += 1
            if counter1 == 255:
                break

        print(f'Channel {channel + 1} Decrypted.')

    torch.cuda.empty_cache()
    output_image = output_image.to('cuda')
    # Normalize the image for display
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to('cuda')  # Mean for each channel
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to('cuda')
    denormalized_image = (output_image * std) + mean
    denormalized_image = torch.clamp(denormalized_image, 0, 1)
    return denormalized_image
