import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self, latent_dim, x_dim=784, h=128):
        super(Generator, self).__init__()
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

    def forward(self, z):
        return self.G(z)


class Discriminator(nn.Module):
    def __init__(self, x_dim=784, h=128):
        super(Discriminator, self).__init__()

        self.D = nn.Sequential(
            nn.Linear(x_dim, h * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(h * 4, h * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(h * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.D(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device):
    for epoch in range(ARGS.n_epochs):
        for i, (x, _) in enumerate(dataloader):
            bs = x.shape[0]
            x = x.view(bs, -1).to(device)

            # Train Generator
            # ---------------
            z = torch.randn((bs, generator.latent_dim))
            g_z = generator(z)
            d_g_z = torch.log(discriminator(g_z))

            optimizer_G.zero_grad()
            loss_g = -torch.mean(d_g_z)
            loss_g.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            d_x = torch.log(discriminator(x))
            z = torch.randn((bs, generator.latent_dim))
            g_z = generator(z)
            d_g_z = torch.log(1 - discriminator(g_z))

            optimizer_D.zero_grad()
            loss_d = - torch.mean(d_x) - torch.mean(d_g_z)
            loss_d.backward()
            optimizer_D.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % ARGS.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(gen_imgs[:25],
                #            'images/{}.png'.format(batches_done),
                #            nrow=5, normalize=True)
                pass


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
    generator = Generator(latent_dim=ARGS.latent_dim)
    discriminator = Discriminator()
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
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    ARGS = parser.parse_args()

    main()
