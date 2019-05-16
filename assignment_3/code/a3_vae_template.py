import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, x_dim, hidden_dim=500, z_dim=20):
        super().__init__()
        self.h = nn.Linear(x_dim, hidden_dim)
        self.μ = nn.Linear(hidden_dim, z_dim)
        self.σ = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        h = torch.tanh(self.h(x))
        μ, σ = self.μ(h), self.σ(h)
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

    def __init__(self, x_dim=784, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(x_dim, hidden_dim, z_dim)
        self.decoder = Decoder(x_dim, hidden_dim, z_dim)

    def forward(self, x):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        μ, σ = self.encoder(x)
        ε = torch.zeros(μ.shape).normal_()
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
        sampled_ims, im_means = None, None
        raise NotImplementedError()

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = []
    for batch in data:
        batch = batch.view(batch.shape[0], -1)
        elbo = model.forward(batch)

        if model.training:
            model.zero_grad()
            elbo.backward()
            optimizer.step()
        average_epoch_elbo.append(elbo.item())

    return np.mean(average_epoch_elbo)


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

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


def main():
    data = bmnist(batch_size=ARGS.batch_size)[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        (train_elbo, val_elbo) = run_epoch(model, data, optimizer)
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo:.4e} val_elbo: {val_elbo:.4e}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')

    ARGS = parser.parse_args()

    main()
