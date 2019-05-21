import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mim
from torchvision.utils import make_grid
import numpy as np
import math
import os
import scipy.stats as stats

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, x_dim, hidden_dim=500, z_dim=20):
        super().__init__()
        self.h = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.Tanh()
        )
        self.μ = nn.Linear(hidden_dim, z_dim)
        self.σ = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        h = self.h(x)
        μ = self.μ(h)
        σ = torch.sqrt(self.σ(h))
        return μ, σ


class Decoder(nn.Module):

    def __init__(self, x_dim, hidden_dim=500, z_dim=20):
        super().__init__()
        self.y = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, x_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        return self.y(z)


class VAE(nn.Module):

    def __init__(self, x_dim, hidden_dim=500, z_dim=20, device='cpu'):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(x_dim, hidden_dim, z_dim)
        self.decoder = Decoder(x_dim, hidden_dim, z_dim)
        self.device = device
        self.to(device)

    def forward(self, x):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        μ, σ = self.encoder(x)
        ε = torch.randn_like(μ, device=self.device)
        z = σ * ε + μ

        y = self.decoder(z)

        l_reg = 0.5 * (σ**2 + μ**2 - 1 - torch.log(σ**2))
        l_recon = x * y.log() + (1 - x) * (1 - y).log()
        elbo = l_reg.sum(dim=-1) - l_recon.sum(dim=-1)
        return elbo.mean()

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        ε = torch.zeros(n_samples, self.z_dim, device=self.device).normal_()
        with torch.no_grad():
            μ = self.decoder(ε)
        img = torch.rand_like(μ, device=self.device).cpu() > μ.cpu()
        return img, μ

    def manifold(self, n):
        xy = np.mgrid[0:n, 0:n].reshape((2, n**2)).T / (n - 1)
        xy = (xy + 4.45e-2) * 9e-1

        z = torch.tensor(stats.norm.ppf(xy), device=self.device, dtype=torch.float)
        with torch.no_grad():
            μ = self.decoder(z)
        return μ


def epoch_iter(model, data, optimizer, device):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = []
    for batch in data:
        batch = batch.view(batch.shape[0], -1).to(device)
        elbo = model.forward(batch)

        if model.training:
            model.zero_grad()
            elbo.backward()
            optimizer.step()
        average_epoch_elbo.append(elbo.item())

    return np.mean(average_epoch_elbo)


def run_epoch(model, data, optimizer, device):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer, device)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer, device)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def save_sample(sample, imw, epoch, nrow=8, slug='sample'):
    sample = sample.view(-1, 1, imw, imw).cpu()
    sample = make_grid(sample, nrow=nrow).numpy().astype(np.float).transpose(1, 2, 0)
    mim.imsave(f"figures/vae_{slug}_{epoch}.png", sample)


def main():
    # general setup
    os.makedirs('./figures', exist_ok=True)
    imw = 28
    device = torch.device(ARGS.device)

    data = bmnist(root=ARGS.data, batch_size=ARGS.batch_size, download=False)[:2]  # ignore test split
    model = VAE(x_dim=imw**2, z_dim=ARGS.zdim, device=device)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        (train_elbo, val_elbo) = run_epoch(model, data, optimizer, device=device)
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo:.4e} val_elbo: {val_elbo:.4e}")

        _, μ_sample = model.sample(ARGS.samples)
        save_sample(μ_sample, imw, epoch)

        if ARGS.zdim == 2:
            manifold = model.manifold(ARGS.manifold_samples)
            save_sample(manifold, imw, epoch, ARGS.manifold_samples, 'manifold')

    torch.save({'train': train_curve, 'val': val_curve}, f"vae_{ARGS.zdim}_curves.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--samples', type=int, default=16,
                        help="How many samples to sample when we sample.")
    parser.add_argument('--manifold_samples', type=int, default=30)
    parser.add_argument('--data', type=str, default='./data',
                        help="DATA dir root")

    ARGS = parser.parse_args()

    main()
