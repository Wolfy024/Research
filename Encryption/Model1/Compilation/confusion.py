import torch
from torchvision import transforms
import numpy as np
from PIL import Image


def confuse(img, key1, key2, key3):
    # Convert the image to a tensor and maintain the format [C, H, W]
    img = img.view(3, -1)  # Flatten H, W into a single dimension
    print(f"Image shape: {img.shape}")  # Should be [3, 65536]
    print(f"Key1 shape: {key1.shape}")  # Should be [65536]

    real_img = img.clone()  # Create a copy of the original image
    for i in range(3):  # Loop over color channels
        if i == 0:
            img[i] = real_img[i, key1]  # Use key1 for the first channel
        elif i == 1:
            img[i] = real_img[i, key2]  # Use key2 for the second channel
        elif i == 2:
            img[i] = real_img[i, key3]  # Use key3 for the third channel
        print(f'Channel {i + 1} done.')

    img = img.view(3, 256, 256)  # Reshape back to [C, H, W]
    return img  # Return the tensor, not the PIL image


def reverse_confuse(img, key1, key2, key3):
    img = img.view(3, -1)  # Flatten H, W into a single dimension
    print(f"Reversed Image shape: {img.shape}")  # Should be [3, 65536]

    real_img = img.clone()  # Create a copy of the scrambled image
    for i in range(3):  # Loop over color channels
        reverse_key = torch.empty_like(key1)
        if i == 0:
            reverse_key[key1] = torch.arange(0, len(key1), dtype=key1.dtype)  # Use key1 for the first channel
            img[i] = real_img[i, reverse_key]
        elif i == 1:
            reverse_key[key2] = torch.arange(0, len(key2), dtype=key2.dtype)  # Use key2 for the second channel
            img[i] = real_img[i, reverse_key]
        elif i == 2:
            reverse_key[key3] = torch.arange(0, len(key3), dtype=key3.dtype)  # Use key3 for the third channel
            img[i] = real_img[i, reverse_key]
        print(f'Reversed Channel {i + 1} done.')

    img = img.view(3, 256, 256)  # Reshape back to [C, H, W]
    return img


def compare_images(original_img, unscrambled_img):
    difference = torch.abs(original_img - unscrambled_img)
    total_difference = torch.sum(difference)
    if total_difference == 0:
        print("The original and unscrambled images are identical!")
    else:
        print(f"There are differences between the images. Total pixel difference: {total_difference.item()}.")


if __name__ == "__main__":
    img = torch.rand(3, 256, 256)
    key1 = torch.randperm(256 * 256, dtype=torch.int)  # Generates a random permutation
    key2 = torch.randperm(256 * 256, dtype=torch.int)
    key3 = torch.randperm(256 * 256, dtype=torch.int)
    scrambled_img = confuse(img, key1, key2, key3)
    unscrambled_img = reverse_confuse(scrambled_img, key1, key2, key3)
    compare_images(img, unscrambled_img)
