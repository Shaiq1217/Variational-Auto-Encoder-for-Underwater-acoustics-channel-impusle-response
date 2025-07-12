import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os


def compute_mse(model, data_loader, device='cpu'):
    model.eval()
    total_mse = 0.0
    num_samples = 0

    with torch.no_grad():
        for x_batch in data_loader:
            x_batch = x_batch.to(device)
            reconstructed, mu, logvar = model(x_batch) 
            batch_mse = F.mse_loss(reconstructed, x_batch, reduction='sum')
            total_mse += batch_mse.item()
            num_samples += x_batch.size(0)

    avg_mse = total_mse / num_samples
    return avg_mse


def plot_multiple(model, data_loader, device='cpu', num_samples=4):
    model.eval()
    total_mse = 0.0
    with torch.no_grad():
        for x_batch in data_loader:
            x_batch = x_batch.to(device)
            recon_batch, _, _ = model(x_batch)

            x_batch = x_batch.cpu().numpy()
            recon_batch = recon_batch.cpu().numpy()

            fig, axs = plt.subplots(num_samples, 2, figsize=(14, 3 * num_samples))

            for i in range(num_samples):
                orig_real = x_batch[i][0].flatten()
                orig_imag = x_batch[i][1].flatten()
                recon_real = recon_batch[i][0].flatten()
                recon_imag = recon_batch[i][1].flatten()
                complex_orig = orig_real + 1j * orig_imag
                complex_recon = recon_real + 1j * recon_imag

                total_mse += np.mean((np.abs(complex_orig) - np.abs(complex_recon)) ** 2)

                axs[i, 0].plot(np.abs(complex_orig), label=f'Original CIR {i}')
                axs[i, 0].set_title(f"Original CIR {i}")
                axs[i, 0].legend()
                axs[i, 0].grid(True)
                
                

                axs[i, 1].plot(np.abs(complex_recon), label=f'Recon CIR {i}', color='orange')
                axs[i, 1].set_title(f"Recon CIR {i}")
                axs[i, 1].legend()
                axs[i, 1].grid(True)
                
                

            plt.suptitle('Original and Reconstructed CIRs', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join("out", "cir_reconstruction_multiple.png"))
            plt.show()
            print(f"Batch MSE: {total_mse / num_samples:.4f}")
            break  # only show first batch
