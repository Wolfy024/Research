# C:\Users\viraj\PycharmProjects\Research\data
from Encryption.Model2.UNET.Model import UNET
import torch


def encode_image(img_path: str) -> torch.Tensor:
    from PIL import Image
    import torchvision.transforms as transforms
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])
    img = transform(img).unsqueeze(0)
    model = UNET()
    model.load_state_dict(torch.load(r"C:\Users\viraj\PycharmProjects\Research\Encryption\Model2\UNET\Models\Epoch_17_UNET_loss_0.12032290241768351.pt"))
    output = model(img)
    torch.save(output, 'output_tuple.pt')
    print("Encrypted image saved as output_tuple.pt")
    return output

