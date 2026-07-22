import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class Data(Dataset):
    def __init__(self, folder1, transforms=None):
        self.folder = folder1
        self.data = os.listdir(folder1)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path_1 = os.path.join(self.folder, self.data[idx])
        img_path_2 = os.path.join(self.folder, self.data[(idx + 1) % len(self.data)])  # Corrected index wrapping
        image_1 = Image.open(img_path_1).convert("RGB")
        image_2 = Image.open(img_path_2).convert("RGB")

        if self.transforms:
            image_1 = self.transforms(image_1)
            image_2 = self.transforms(image_2)

        return image_1, image_2


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Example transform
        transforms.ToTensor()
    ])
    data = Data(r"C:\Users\viraj\PycharmProjects\Research\Watermark\Datasets\Animals", transforms=transform)
    data = DataLoader(data, batch_size=16, shuffle=True)
    print(len(data))
