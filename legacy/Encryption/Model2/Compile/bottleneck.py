# C:\Users\viraj\PycharmProjects\Research\data
from Encryption.Model2.UNET.Model import UNET
import torch


def get_bottleneck(img_path: str) -> tuple:
    from PIL import Image
    import torchvision.transforms as transforms
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])
    img = transform(img).unsqueeze(0).to('cuda')
    model = UNET().to('cuda')
    model.load_state_dict(torch.load(r"C:\Users\viraj\PycharmProjects\Research\Encryption\Model2\UNET\Models\Epoch_17_UNET_loss_0.12032290241768351.pt"))
    output, output1, output2, output3 = model(img, bottleneck=True)
    return output, output3, output2, output1

get_bottleneck(r"C:\Users\viraj\PycharmProjects\Research\data\Abyssinian_1.jpg")