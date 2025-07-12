from src.cir import save_reshaped_cir, plot_cir
from src.dataloader import load_data
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from src.model import ConvVAE
from src.training import train_vae
import torch
from sklearn.decomposition import PCA
from src.metrics import compute_mse, plot_multiple

def preprocess_data():
    # Saving reshaped CIR
  mat_files = glob.glob(os.path.join("data", "cirmat", "*.mat"))
  out_path = os.path.join("data", "cir")

  for idx, cir in enumerate(mat_files):
    out_file = os.path.join(out_path, f"cir_{idx}.npy")
    save_reshaped_cir(cir, out_file)
    print(f"[✓] Saved: {out_file}")


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
  num_samples = 187  
  z = torch.randn(num_samples, latent_dim).to(device)  

  with torch.no_grad():
    generated = model.decode(z) 
    generated = generated.cpu().numpy()
    magnitudes = []

    for sample in generated:
        real = sample[0]
        imag = sample[1]
        mag = np.abs(real + 1j * imag)
        
        # Flatten each CIR into a 1D vector (optional: mean across one axis)
        flat_mag = mag.flatten()  # shape: (2401,)
        magnitudes.append(flat_mag)

    # Stack into a 2D matrix: (num_samples x taps)
    heatmap_data = np.stack(magnitudes) # shape: (180, 2401)

    # Plot the global heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data ,cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='Magnitude')
    plt.xlabel("Sample index")
    plt.ylabel("Block Index (Time)")
    plt.title("Magnitude Heatmap of All Generated CIRs")
    plt.savefig(os.path.join("out", "gen_cir_heatmap.png"))
    plt.tight_layout()
    plt.show()
  

def visualizetSNE():
    train_loader, val_loader = load_data() 
    device = 'cpu'

    model = ConvVAE(latent_dim=32)
    model.load_state_dict(torch.load(os.path.join("out", "best_vae_model.pth"), map_location=device))  
    model.to(device)
    model.eval()  

    latent_vectors = []

    with torch.no_grad():
        for x_batch in val_loader:
            x_batch = x_batch.to(device)
            mu, logvar = model.encode(x_batch)
            z = model.reparameterize(mu, logvar)
            latent_vectors.append(z.cpu())
    
    z_all = torch.cat(latent_vectors, dim=0).numpy()

    # Apply t-SNE
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_all)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.7, s=40, edgecolors='k')
    plt.title("t-SNE of VAE Latent Space (Encoded CIRs)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(os.path.join("out", "tsne_latent_cir_nolabel.png"))
    plt.show()

def getMSE():
  train_loader, val_loader = load_data() 
  device = 'cpu'
  model = ConvVAE(latent_dim=32)
  model.load_state_dict(torch.load(os.path.join("out" , "best_vae_model.pth"), map_location=device))  
  model.to(device)
  model.eval()  
  plot_multiple(model, val_loader, device=device, num_samples=4)
  

if __name__ == "__main__":
  # visualizetSNE()
  getMSE()