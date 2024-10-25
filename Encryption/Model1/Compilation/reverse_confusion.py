import torch
from torchvision import transforms
import numpy as np
from PIL import Image


def reverse_confuse(img, key1, key2, key3):
    # Convert the image to a tensor and flatten it [C, H * W]
    img = transforms.ToTensor()(img).view(3, -1)
    print(f"Reversed Image shape: {img.shape}")  # Should be [3, 65536]
    real_img = img.clone()  # Create a copy of the original scrambled image
    for i in range(3):  # Loop over color channels
        for j in range(256 * 256):
            img[i, j] = real_img[i, key1[j]]

        if i == 0:
            key1 = key2
            print('Channel 1 reversed.')
        elif i == 1:
            key1 = key3
            print('Channel 2 reversed.')

    img = img.view(3, 256, 256)  # Reshape back to [C, H, W]
    # Convert back to PIL image after reversing the scrambling
    img = transforms.ToPILImage()(img)
    return img


if __name__ == "__main__":
    img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
    key1 = torch.randint(0, 256, (256, 256), dtype=torch.int).flatten()
    key2 = torch.randint(0, 256, (256, 256), dtype=torch.int).flatten()
    key3 = torch.randint(0, 256, (256, 256), dtype=torch.int).flatten()
    reversed_img = reverse_confuse(img, key1, key2, key3)
    reversed_img.show()
