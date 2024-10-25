def decrypt(image):
    import numpy as np
    from torchvision import transforms
    import decrypt_channels, xor, reverse_confusion
    import Generate_Sequences
    import torch

    # Generate confusion keys
    key1 = Generate_Sequences.generate_confusion_keys(256 * 256, 452343456).flatten()
    key2 = Generate_Sequences.generate_confusion_keys(256 * 256, 567876543).flatten()
    key3 = Generate_Sequences.generate_confusion_keys(256 * 256, 34567876543).flatten()

    # Reverse confusion
    final_img = reverse_confusion.reverse_confuse(image, key3, key2, key1)
    final_img = transforms.ToTensor()(final_img).view(3, 256, 256).to('cuda')
    final_img = final_img.int()
    data = torch.round(torch.abs(Generate_Sequences.generate_xor_keys(2.6543))).int().view(3, 256, 256).to('cuda')
    xor_result = xor.xor_image_tensor(final_img, data)
    xor_tensor = transforms.ToTensor()(xor_result)
    final_img = (xor_tensor / 255)
    # Decrypt channels
    transformed_image = decrypt_channels.decrypt_channels(0.244312, 0.314524, 0.14256, 12.124124, final_img)
    transformed_image = transforms.ToPILImage()(transformed_image.cpu())
    print('1')
    transformed_image.show()
