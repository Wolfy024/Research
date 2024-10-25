import encrypt_channels, confusion, decrypt_channels
import Generate_Sequences
from torchvision import transforms
from PIL import Image
from torchvision import transforms

# if __name__ == "__main__":
#     key1 = Generate_Sequences.generate_confusion_keys(256 * 256, 452343456).flatten()
#     key2 = Generate_Sequences.generate_confusion_keys(256 * 256, 567876543).flatten()
#     key3 = Generate_Sequences.generate_confusion_keys(256 * 256, 34567876543).flatten()
#     # transformed_image = encrypt_channels.encrypt_channels(0.244312, 0.314524, 0.14256, 12.124124, r"C:\Users\viraj\PycharmProjects\Research\data\Abyssinian_29.jpg")
#     # transformed_image = transformed_image.to('cuda')
#     # final_img = confusion.confuse(transformed_image, key1, key2, key3)
#     # transforms.ToPILImage()(final_img.cpu()).save(r"C:\Users\viraj\PycharmProjects\Research\Encryption\Compilation\encrypted.png")
#     final_img = Image.open('encrypted.png')
#     final_img = transforms.ToTensor()(final_img)
#     final_img = confusion.reverse_confuse(final_img, key1, key2, key3)
#     final_img = final_img.to('cuda')
#     transformed_image = decrypt_channels.decrypt_channels(0.244312, 0.314524, 0.14256, 12.124124, final_img)
#     transformed_image = transforms.ToPILImage()(transformed_image.cpu())
#     transformed_image.show()

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


def encrypt(img, save_img_name, width=256, height=256):
    channel_confusion_1_output = Generate_Sequences.generate_chaotic_sequences(channel_confusion_key1_input,
                                                                               channel_confusion1_seed_input)
    channel_confusion_2_output = Generate_Sequences.generate_chaotic_sequences(channel_confusion_key2_input,
                                                                               channel_confusion2_seed_input)
    channel_confusion_3_output = Generate_Sequences.generate_chaotic_sequences(channel_confusion_key3_input,
                                                                               channel_confusion3_seed_input)
    encoder_1_output = Generate_Sequences.generate_keys(encoder1_key_input, encoder1_seed_input)

    key_1 = Generate_Sequences.generate_confusion_keys(width * height, position_confusion1_seed_input).flatten()
    key_2 = Generate_Sequences.generate_confusion_keys(width * height, position_confusion2_seed_input).flatten()
    key_3 = Generate_Sequences.generate_confusion_keys(width * height, position_confusion3_seed_input).flatten()

    transformed_image = encrypt_channels.encrypt_channels(encoder_1_output,
                                                          channel_confusion_1_output,
                                                          channel_confusion_2_output,
                                                          channel_confusion_3_output,
                                                          img)
    transformed_image = transformed_image.to('cuda')
    final_img = confusion.confuse(transformed_image,
                                  key_1,
                                  key_2,
                                  key_3)
    transforms.ToPILImage()(final_img.cpu()).save(
        fr"C:\Users\viraj\PycharmProjects\Research\Encryption\Model1\Compilation\{save_img_name}.png")


def decrypt(img, save_img_name, width=256, height=256):
    img = Image.open(img)
    img = transforms.ToTensor()(img).view(3, 256, 256)
    channel_confusion_1_output = Generate_Sequences.generate_chaotic_sequences(channel_confusion_key1_input,
                                                                               channel_confusion1_seed_input)
    channel_confusion_2_output = Generate_Sequences.generate_chaotic_sequences(channel_confusion_key2_input,
                                                                               channel_confusion2_seed_input)
    channel_confusion_3_output = Generate_Sequences.generate_chaotic_sequences(channel_confusion_key3_input,
                                                                               channel_confusion3_seed_input)
    encoder_1_output = Generate_Sequences.generate_keys(encoder1_key_input, encoder1_seed_input)

    key_1 = Generate_Sequences.generate_confusion_keys(width * height, position_confusion1_seed_input).flatten()
    key_2 = Generate_Sequences.generate_confusion_keys(width * height, position_confusion2_seed_input).flatten()
    key_3 = Generate_Sequences.generate_confusion_keys(width * height, position_confusion3_seed_input).flatten()

    transformed_image = decrypt_channels.decrypt_channels(encoder_1_output,
                                                          channel_confusion_1_output,
                                                          channel_confusion_2_output,
                                                          channel_confusion_3_output,
                                                          img)
    transformed_image = transformed_image.to('cuda')
    final_img = confusion.reverse_confuse(transformed_image,
                                  key_1,
                                  key_2,
                                  key_3)

    transforms.ToPILImage()(final_img.cpu()).save(
        fr"C:\Users\viraj\PycharmProjects\Research\Encryption\Model1\Compilation\{save_img_name}.png")


encrypt(r"C:\Users\viraj\PycharmProjects\Research\data\Abyssinian_29.jpg", 'wtf')
decrypt('wtf', 'wtff')