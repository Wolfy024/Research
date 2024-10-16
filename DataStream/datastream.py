from torch.utils.data import Dataset


class DataStream(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, transform=False):
        if self.transform:
            return self.transform(self.data[idx]), self.targets[idx]
        else:
            return self.data[idx], self.targets[idx]
