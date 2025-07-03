import numpy as np
from torch.utils.data import Dataset
import torch

class CIRDataloader(Dataset):
    def __init__(self, cir_array, transform=None):
        self.cir_array = torch.Tensor(cir_array)
        self.transform = transform

    def __len__(self):
        return self.cir_array.shape[0]

    def __getitem__(self, idx):
        sample = self.cir_array[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample