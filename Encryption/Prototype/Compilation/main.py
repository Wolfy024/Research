import torch
from torchvision import transforms
import Generate_Sequences
import encrypt_channels
import decrypt_channels
import confusion
import OpenEXR
import Imath
import numpy as np
import torch


def save_img(filename, image_tensor):
    # Ensure the input tensor is in the correct shape and type
    if image_tensor.dim() != 3 or image_tensor.size(0) != 3:
        raise ValueError("Input tensor must be of shape (3, height, width)")

    # Get the height and width from the tensor shape
    height, width = image_tensor.size(1), image_tensor.size(2)

    # Create an OpenEXR file header
    header = OpenEXR.Header(width, height)
    channels = ["R", "G", "B"]

    # Create a dictionary for the pixel data
    pixel_data = {}

    # Prepare pixel data for each channel and ensure they are in float format
    for i, channel in enumerate(channels):
        pixel_data[channel] = image_tensor[i].cpu().detach().numpy().astype(np.float32).tobytes()  # Convert to bytes

    # Define channels in the header
    for channel in channels:
        header['channels'][channel] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))

    # Open the EXR file and write the pixels
    exr_file = OpenEXR.OutputFile(filename, header)
    exr_file.writePixels(pixel_data)
    exr_file.close()  # Explicitly close the file


def load_img(filename):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(filename)

    # Read the header to get image dimensions
    header = exr_file.header()
    width = header['dataWindow'].max.x + 1
    height = header['dataWindow'].max.y + 1

    # Prepare to read pixel data
    channels = ["R", "G", "B"]
    pixel_data = {}

    for channel in channels:
        # Read the pixel data for each channel
        pixel_data[channel] = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))

    # Close the EXR file
    exr_file.close()

    # Convert pixel data to a NumPy array and reshape it
    image_array = np.zeros((3, height, width), dtype=np.float32)
    for i, channel in enumerate(channels):
        # Convert bytes back to a numpy array
        image_array[i] = np.frombuffer(pixel_data[channel], dtype=np.float32).reshape((height, width))

    # Convert to a PyTorch tensor
    image_tensor = torch.from_numpy(image_array)

    return image_tensor


# if __name__ == "__main__":
#     key1 = Generate_Sequences.generate_confusion_keys(256 * 256, 452343456).flatten()
#     key2 = Generate_Sequences.generate_confusion_keys(256 * 256, 567876543).flatten()
#     key3 = Generate_Sequences.generate_confusion_keys(256 * 256, 34567876543).flatten()
#     encoder1 = Generate_Sequences.generate_keys(2, 1)
#     confusion_key1 = Generate_Sequences.generate_chaotic_sequences(1, 1)
#     confusion_key2 = Generate_Sequences.generate_chaotic_sequences(2, 1)
#     confusion_key3 = Generate_Sequences.generate_chaotic_sequences(3, 1)
#     # Encrypt and save as EXR
#     transformed_image = encrypt_channels.encrypt_channels(encoder1, confusion_key1, confusion_key2, confusion_key3,
#                                                           r"C:\Users\viraj\PycharmProjects\Research\data\Abyssinian_20.jpg")
#     transformed_image = transformed_image.to('cuda')
#     final_img = confusion.confuse(transformed_image, key1, key2, key3)
#     save_img("output.exr", final_img)
#     final_img = load_img("output.exr")
#     final_img = final_img.to('cuda')
#     loaded_image = confusion.reverse_confuse(final_img, key1, key2, key3)
#     decrypted_image = decrypt_channels.decrypt_channels(encoder1, confusion_key1, confusion_key2, confusion_key3,
#                                                         loaded_image)
#     # Convert decrypted tensor to PIL image and display
#     decrypted_image_pil = transforms.ToPILImage()(decrypted_image.cpu())
#     decrypted_image_pil.save(r"C:\Users\viraj\PycharmProjects\Research\Encryption\Prototype\Compilation\decrypted.png")
#     decrypted_image_pil.show()

channel_confusion_key1_input = float(input("Enter key."))
channel_confusion_key2_input = float(input("Enter key."))
channel_confusion_key3_input = float(input("Enter key."))
channel_confusion1_seed_input = int(input("Enter seed. PSNR"))
channel_confusion2_seed_input = int(input("Enter seed. PSNR"))
channel_confusion3_seed_input = int(input("Enter seed. PSNR"))
position_confusion1_seed_input = int(input("Enter seed. PSNR"))
position_confusion2_seed_input = int(input("Enter seed. PSNR"))
position_confusion3_seed_input = int(input("Enter seed. PSNR"))
encoder1_key_input = float(input("Enter key."))
encoder1_seed_input = int(input("Enter seed."))
encoder2_key_input = float(input("Enter key."))
encoder2_seed_input = int(input("Enter seed."))
encoder3_key_input = float(input("Enter key."))
encoder3_seed_input = int(input("Enter seed."))


def encrypt(img, save_img_name, width=256, height=256):
    channel_confusion_1_output = Generate_Sequences.generate_chaotic_sequences(channel_confusion_key1_input,
                                                                               channel_confusion1_seed_input)
    channel_confusion_2_output = Generate_Sequences.generate_chaotic_sequences(channel_confusion_key2_input,
                                                                               channel_confusion2_seed_input)
    channel_confusion_3_output = Generate_Sequences.generate_chaotic_sequences(channel_confusion_key3_input,
                                                                               channel_confusion3_seed_input)
    encoder_1_output = Generate_Sequences.generate_keys(encoder1_key_input, encoder1_seed_input)
    encoder_2_output = Generate_Sequences.generate_keys(encoder2_key_input, encoder2_seed_input)
    encoder_3_output = Generate_Sequences.generate_keys(encoder3_key_input, encoder3_seed_input)

    key_1 = Generate_Sequences.generate_confusion_keys(width * height, position_confusion1_seed_input).flatten()
    key_2 = Generate_Sequences.generate_confusion_keys(width * height, position_confusion2_seed_input).flatten()
    key_3 = Generate_Sequences.generate_confusion_keys(width * height, position_confusion3_seed_input).flatten()

    transformed_image = encrypt_channels.encrypt_channels(encoder_1_output,
                                                          encoder_2_output,
                                                          encoder_3_output,
                                                          channel_confusion_1_output,
                                                          channel_confusion_2_output,
                                                          channel_confusion_3_output,
                                                          img)
    transformed_image = transformed_image.to('cuda')
    final_img = confusion.confuse(transformed_image,
                                  key_1,
                                  key_2,
                                  key_3)
    save_img(fr"/Encryption/Prototype\Compilation\Imagess\{save_img_name}.exr", final_img)


def decrypt(img, save_img_name, width=256, height=256):
    img = load_img(img)
    img = img.to('cuda')
    channel_confusion_1_output = Generate_Sequences.generate_chaotic_sequences(channel_confusion_key1_input,
                                                                               channel_confusion1_seed_input)
    channel_confusion_2_output = Generate_Sequences.generate_chaotic_sequences(channel_confusion_key2_input,
                                                                               channel_confusion2_seed_input)
    channel_confusion_3_output = Generate_Sequences.generate_chaotic_sequences(channel_confusion_key3_input,
                                                                               channel_confusion3_seed_input)
    encoder_1_output = Generate_Sequences.generate_keys(encoder1_key_input, encoder1_seed_input)
    encoder_2_output = Generate_Sequences.generate_keys(encoder2_key_input, encoder2_seed_input)
    encoder_3_output = Generate_Sequences.generate_keys(encoder3_key_input, encoder3_seed_input)
    key_1 = Generate_Sequences.generate_confusion_keys(width * height, position_confusion1_seed_input).flatten()
    key_2 = Generate_Sequences.generate_confusion_keys(width * height, position_confusion2_seed_input).flatten()
    key_3 = Generate_Sequences.generate_confusion_keys(width * height, position_confusion3_seed_input).flatten()
    final_img = confusion.reverse_confuse(img,
                                  key_1,
                                  key_2,
                                  key_3)
    transformed_image = decrypt_channels.decrypt_channels(encoder_1_output,
                                                          encoder_2_output,
                                                          encoder_3_output,
                                                          channel_confusion_1_output,
                                                          channel_confusion_2_output,
                                                          channel_confusion_3_output,
                                                          final_img)
    transforms.ToPILImage()(transformed_image.cpu()).save(fr"C:\Users\viraj\PycharmProjects\Research\Encryption\Model1\Compilation\Imagess\{save_img_name}.png")


encrypt(r"C:\Users\viraj\PycharmProjects\Research\data\Abyssinian_29.jpg", 'wtf')
decrypt(r'C:\Users\viraj\PycharmProjects\Research\Encryption\Model1\Compilation\Imagess\wtf.exr', 'wtff')
