if __name__ == "__main__":
    import encrypt_channels, xor, confusion, decrypt, decrypt_channels
    import Generate_Sequences
    import torch
    from PIL import Image
    from torchvision import transforms
    key1 = Generate_Sequences.generate_confusion_keys(256 * 256, 452343456).flatten()
    key2 = Generate_Sequences.generate_confusion_keys(256 * 256, 567876543).flatten()
    key3 = Generate_Sequences.generate_confusion_keys(256 * 256, 34567876543).flatten()
    transformed_image = encrypt_channels.encrypt_channels(0.244312, 0.314524, 0.14256, 12.124124, r"C:\Users\viraj\PycharmProjects\Research\data\Abyssinian_29.jpg")
    transformed_image = transformed_image.to('cuda')
    final_img = confusion.confuse(transformed_image, key1, key2, key3)
    transforms.ToPILImage()(final_img.cpu()).save(r"C:\Users\viraj\PycharmProjects\Research\Encryption\Compilation\encrypted.png")
    final_img = confusion.reverse_confuse(final_img, key1, key2, key3)
    final_img = final_img.to('cuda')
    transformed_image = decrypt_channels.decrypt_channels(0.244312, 0.314524, 0.14256, 12.124124, final_img)
    transformed_image = transforms.ToPILImage()(transformed_image.cpu())
    transformed_image.show()
