import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from Generate_Sequences import generate_keys, generate_chaotic_sequences
from encrypt_channels import encrypt_channels
from decrypt_channels import decrypt_channels


if __name__ == "__main__":
    transformed_image_cpu = encrypt_channels(0.91354, 0.38435, 5.134545, 4.4565)
    denormalized_image = transformed_image_cpu.cpu()  # Move to CPU for display
    to_pil = transforms.ToPILImage()
    image_from_tensor = to_pil(denormalized_image)
    image_from_tensor.show()
    decrypted_image_cpu = decrypt_channels(0.91354, 0.38435, 5.134545, 4.4565, transformed_image_cpu)
    decrypted_image_from_tensor = to_pil(decrypted_image_cpu.cpu())
    decrypted_image_from_tensor.show()
