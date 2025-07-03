import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1),  # (16, 25, 25)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (32, 13, 13)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (64, 7, 7)
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1, padding=1),  # (32, 13, 13)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1, padding=1),  # (16, 25, 25)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 2, 3, stride=2, output_padding=1, padding=1),  # (2, 49, 49)
            nn.Tanh()  # Output range [-1, 1]
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z).view(-1, 64, 7, 7)
        x = self.decoder(x)

        # Center crop to 49x49
        crop = (x.size(-1) - 49) // 2
        x = x[:, :, crop:crop+49, crop:crop+49]
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    


