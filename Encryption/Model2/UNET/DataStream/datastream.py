import os
from PIL import Image
from torch.utils.data import Dataset


class DataStream(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data = os.listdir(data_folder)
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_folder, self.data[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)  # Now this will correctly apply the transform
        return image
