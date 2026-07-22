# C:\Users\viraj\PycharmProjects\Research\data
from Encryption.Model2.UNET.Model import UNET
import torch
from torchvision import transforms
from Encryption.Model2.Encoder.swap_tensors import decrypt


def get_reverse(x):
    from PIL import Image
    import torchvision.transforms as transforms
    x = torch.load(x)
    model = UNET().to('cuda')
    model.load_state_dict(torch.load(
        r"C:\Users\viraj\PycharmProjects\Research\Encryption\Model2\UNET\Models\Epoch_17_UNET_loss_0"
        r".12032290241768351.pt"))
    # keys = ['x', 'x1', 'x2', 'x3', 'bigkey']
    keys = [3653233072, 741159964, 3234978105, 3627292614, 'oODE8oyR8TsEmICEoGMDZoc8g--d3m9-q0bxEusKrJM=']
    x = decrypt(x[0], x[1], x[2], x[3], keys)
    output = model(x=x[0], x1=x[1], x2=x[2], x3=x[3], reverse=True)
    print(output.shape)
    output = output.squeeze(0)
    transformation = transforms.ToPILImage()
    output = transformation(output)
    output.show()


if __name__ == "__main__":
    get_reverse(r'C:\Users\viraj\PycharmProjects\Research\Encryption\Model2\Compile\decrypted_model.pt')
