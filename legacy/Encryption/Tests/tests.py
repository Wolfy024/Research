import torch
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from PIL import Image
import numpy as np
import os


def load_image(image_path):
    # Load image as PIL and convert to RGB
    image = Image.open(image_path).convert('RGB')
    return np.array(image)


def calculate_ssim(original_img, decrypted_img):
    # Ensure both images are of the same size
    if original_img.shape != decrypted_img.shape:
        raise ValueError("Images must have the same dimensions for SSIM calculation.")

    # Check if the images are at least 7x7 pixels
    if original_img.shape[0] < 7 or original_img.shape[1] < 7:
        raise ValueError("Images must be at least 7x7 pixels for SSIM calculation.")

    # Calculate SSIM with multichannel=True for RGB images
    return ssim(original_img, decrypted_img, multichannel=True)


def add_salt_pepper_noise(X_img):
    # Need to produce a copy as to not modify the original image
    X_img_copy = X_img.copy()
    row, col, _ = X_img_copy.shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_img_copy.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_img_copy.size * (1.0 - salt_vs_pepper))

    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img_copy.shape]
    X_img_copy[coords[0], coords[1], :] = 1  # White pixels for salt

    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img_copy.shape]
    X_img_copy[coords[0], coords[1], :] = 0  # Black pixels for pepper

    return X_img_copy


def save_image(image_array, output_path):
    # Convert NumPy array back to PIL Image and save
    image = Image.fromarray((image_array * 255).astype(np.uint8))  # Scale back to 0-255 for saving
    image.save(output_path)


if __name__ == "__main__":
    # Load original and decrypted images
    original_img_path = r"C:\Users\viraj\PycharmProjects\Research\Encryption\Model1\Compilation\original.png"
    decrypted_img_path = r"C:\Users\viraj\PycharmProjects\Research\Encryption\Model1\Compilation\decrypted.png"
    encrypted_img_path = r'C:\Users\viraj\PycharmProjects\Research\Encryption\Model1\Compilation\encrypted.png'

    original_img = load_image(original_img_path)
    decrypted_img = load_image(decrypted_img_path)
    encrypted_img = load_image(encrypted_img_path)

    # Calculate SSIM
    try:
        ssim_index = calculate_ssim(original_img, decrypted_img)
    except ValueError as e:
        print(e)
        ssim_index = None  # Handle the error appropriately

    # Add salt-and-pepper noise and save the noisy image
    noisy_img = add_salt_pepper_noise(encrypted_img)

    # Define output path for the noisy image
    noisy_img_path = r"C:\Users\viraj\PycharmProjects\Research\Encryption\Model1\Compilation\noisy_image.png"
    save_image(noisy_img, noisy_img_path)

    # Print SSIM result if it was calculated
    if ssim_index is not None:
        print(f"SSIM: {ssim_index:.4f}")
