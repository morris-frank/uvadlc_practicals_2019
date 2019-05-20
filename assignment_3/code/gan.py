import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from statistics import mean


class Generator(nn.Module):
    def __init__(self, latent_dim, x_dim=784, h=128, device=None):
        super(Generator, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.G = nn.Sequential(
            nn.Linear(self.latent_dim, h),
            nn.LeakyReLU(0.2),
            nn.Linear(h, h * 2),
            nn.BatchNorm1d(h * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(h * 2, h * 4),
            nn.BatchNorm1d(h * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(h * 4, h * 8),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(h * 8),
            nn.Linear(h * 8, x_dim),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, z):
        return self.G(z)

    def sample(self, bs):
        z = torch.randn((bs, self.latent_dim), device=self.device)
        return self.forward(z)


class Discriminator(nn.Module):
    def __init__(self, x_dim=784, h=128, device=None):
        super(Discriminator, self).__init__()

        self.D = nn.Sequential(
            nn.Linear(x_dim, h * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(h * 4, h * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(h * 2, 1),
            nn.Sigmoid()
        )
        self.to(device)

    def forward(self, img):
        return self.D(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device):
    losses_d, losses_g = [], []
    for epoch in range(ARGS.n_epochs):
        _losses_d, _losses_g = [], []
        for i, (x, _) in enumerate(dataloader):
            bs, _, imw, imh = x.shape
            x = x.view(bs, -1).to(device)

            # Train Generator
            # ---------------
            g_z = generator.sample(bs)
            d_g_z = torch.log(discriminator(g_z))

            optimizer_G.zero_grad()
            loss_g = -torch.mean(d_g_z)
            loss_g.backward()
            _losses_g.append(loss_g.item())
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            d_x = torch.log(discriminator(x))
            g_z = generator.sample(bs)
            d_g_z = torch.log(1 - discriminator(g_z))

            optimizer_D.zero_grad()
            loss_d = - torch.mean(d_x) - torch.mean(d_g_z)
            loss_d.backward()
            _losses_d.append(loss_d.item())
            optimizer_D.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % ARGS.save_interval == 0:
                imgs = generator.sample(25).detach().view(25, 1, imw, imh)
                save_image(imgs, f"figures/gan_{batches_done}.png", nrow=5, normalize=True)
        losses_d.append(mean(_losses_d))
        losses_g.append(mean(_losses_g))
    torch.save({'G': losses_g, 'D': losses_d}, 'gan_curves.pt')


def main():
    # Create output image directory
    os.makedirs('figures', exist_ok=True)
    device = torch.device(ARGS.device)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))])),
        batch_size=ARGS.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(latent_dim=ARGS.latent_dim, device=device)
    discriminator = Discriminator(device=device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=ARGS.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=ARGS.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=250,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    ARGS = parser.parse_args()

    main()
