import torch
import glob
import os
from torch.utils.data import Dataset


class MySlideDataset(Dataset):
    """
    One .pt file per slide, each containing a dict with at least
    'features' (n_patches x 1536), 'label' and 'slide_id'.
    """

    def __init__(self, data_path):
        super(MySlideDataset, self).__init__()
        self.bags = sorted(glob.glob(os.path.join(data_path, "*.pt")))
        if not self.bags:
            raise FileNotFoundError(f"No .pt files found in {data_path}")
        self.data = [torch.load(bag, map_location='cpu') for bag in self.bags]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
