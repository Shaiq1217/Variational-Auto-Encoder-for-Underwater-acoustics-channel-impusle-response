import numpy as np
from torch.utils.data import Dataset
import torch
from src.augment import CIRAugment
import glob
import os
from torch.utils.data import DataLoader

def load_data(val_split = 0.15, batch_size = 8):
  data_load = glob.glob(os.path.join("data", "cir", "*.npy"))
  cir_data = np.concatenate([np.load(cir_name) for cir_name in data_load], axis = 0)
    
  print(f"[✓] Loaded {len(data_load)} CIR files. Shape: {cir_data.shape}. Total {np.ceil(cir_data.shape[0] / 8)} batches.")
  dataset = CIRDataloader(cir_data)
  val_size = int(len(dataset) * val_split)
  train_size = len(dataset) - val_size

  train_set = CIRDataloader(cir_data[:train_size], transform=CIRAugment())
  val_set = CIRDataloader(cir_data[train_size:])
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
  print(f"Train loader size: {len(train_loader)} batches")
  print(f"Validation loader size: {len(val_loader)} batches")

  return train_loader, val_loader


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