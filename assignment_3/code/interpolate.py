import torch
from gan import Generator
from argparse import ArgumentParser
from math import *
from torchvision.utils import save_image

DEVICE = torch.device('cpu')


def load_model(fp, latent_dim):
    _state_dict = torch.load(fp, map_location=DEVICE)
    generator = Generator(latent_dim=latent_dim, device=DEVICE)
    generator.load_state_dict(_state_dict)
    return generator


def main(args, n=10, steps=20):
    generator = load_model(args.fp, latent_dim=args.latent_dim)

    while True:
        imgs = torch.zeros(n * steps, 1, 28, 28)
        a = None
        for i in range(n):
            z, a = generator.interpolate(a, steps)
            imgs[i*steps:(i+1)*steps, ...] = z.detach().view(-1, 1, 28, 28)
        save_image(imgs, f"figures/gan_interpolate.png", nrow=steps, normalize=True)
        input()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--fp', type=str, default='gan_mnist_generator.pt')
    ARGS = parser.parse_args()
    main(ARGS)

