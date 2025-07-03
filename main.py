from src.cir import save_reshaped_cir, plot_cir
from src.dataloader import CIRDataloader
from src.augment import CIRAugment
import glob
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from src.model import ConvVAE
from src.training import train_vae
import torch

def preprocess_data():
    # Saving reshaped CIR
  mat_files = glob.glob(os.path.join("data", "cirmat", "*.mat"))
  out_path = os.path.join("data", "cir")

  for idx, cir in enumerate(mat_files):
    out_file = os.path.join(out_path, f"cir_{idx}.npy")
    save_reshaped_cir(cir, out_file)
    print(f"[✓] Saved: {out_file}")

def load_data(val_split = 0.15, batch_size = 8):
  data_load = glob.glob(os.path.join("data", "cir", "*.npy"))
  cir_data = np.concatenate([np.load(cir_name) for cir_name in data_load], axis = 0)
    
  print(f"[✓] Loaded {len(data_load)} CIR files. Shape: {cir_data.shape}")
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

def recon_from_original(loader, device, model):
  
  # Get one sample from val_loader
  for batch in loader:
      original = batch[1].unsqueeze(0).to(device)  # shape: (1, 2, 49, 49)
      break

  with torch.no_grad():
      recon, mu, logvar = model(original)

  # Visualize 1 channel (e.g., real part)
  
  #Plot both side by side
  original_np = original[0].cpu().numpy()  
  complex_original = original_np[0].flatten() + 1j * original_np[1].flatten()

  sample = recon[0].cpu().numpy()  # shape: (2, 49, 49)
  complex_recon = sample[0].flatten() + 1j * sample[1].flatten()

  fig, axes = plt.subplots(1, 2, figsize=(10, 5))
  axes[0].plot(np.abs(complex_original), label='Original CIR')  
  axes[0].set_title('Original CIR')
  axes[0].legend()

  axes[1].plot(np.abs(complex_recon), label='Reconstructed CIR', color='orange')
  axes[1].set_title('Reconstructed CIR')
  axes[1].legend()

  plt.tight_layout()
  plt.savefig(os.path.join("out", "cir_reconstruction_1.png"))
  plt.show()
    


def main():
  train_loader, val_loader = load_data() 
  device = 'cpu'

  model = ConvVAE(latent_dim=32)
  model.load_state_dict(torch.load(os.path.join("out" , "best_vae_model.pth"), map_location=device))  
  model.to(device)
  model.eval()  

  latent_dim = 32 
  num_samples = 4  
  z = torch.randn(num_samples, latent_dim).to(device)  

  with torch.no_grad():
    generated = model.decode(z) 
  for idx, sample in enumerate(generated):
    sample = sample.cpu().numpy()
    sample = sample[0].flatten() + 1j * sample[1].flatten()
    plt.subplot(2, 2, idx + 1)
    plt.plot(np.abs(sample), label='Generated CIR')
    plt.title(f"Generated CIR #{idx+1}")
    plt.xlabel("Tap Index")
    plt.ylabel("Amplitude")
    plt.grid(True)

 
  plt.savefig(os.path.join("out", "generated_cirs.png"))
  plt.tight_layout()
  plt.show()

  

if __name__ == "__main__":
  main()