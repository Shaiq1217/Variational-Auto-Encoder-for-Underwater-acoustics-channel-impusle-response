import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

def vae_loss(recon, x, mu, logvar):
      recon_loss = F.mse_loss(recon, x, reduction='sum')
      kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
      return recon_loss + kl

def train_vae(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            batch = batch.to(device)  
            optimizer.zero_grad()

            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss = vae_loss(recon, batch, mu, logvar)
                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        # Save model checkpoint
        if val_loss == min(val_losses):
            torch.save(model.state_dict(), 'best_vae_model.pth')
            print(f"Saved best model at epoch {epoch}")
        # Write losses to csv
        with open(os.path.join('out', 'losses.csv'), 'a') as f:
            f.write("epoch,train_loss,val_loss\n") if epoch == 1 else None
            f.write(f"{epoch},{train_loss},{val_loss}\n")

    print("Training complete.")
    return train_losses, val_losses
