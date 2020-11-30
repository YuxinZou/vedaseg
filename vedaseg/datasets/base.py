from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """ BaseDataset
    """
    CLASSES = None

    PALETTE = None

    def __init__(self, transform=None):
        self.transform = transform

    def process(self, data):
        if self.transform:
            augmented = self.transform(data)
            return augmented['image'], augmented['mask']
        else:
            return data['image'], data['mask']
