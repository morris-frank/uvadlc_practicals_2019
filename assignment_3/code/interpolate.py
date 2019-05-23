import torch
from gan import Generator as GAN_Generator
from nf import Model as NF_Model
from argparse import ArgumentParser
from math import *
from torchvision.utils import save_image

DEVICE = torch.device('cpu')


def load_model(fp, latent_dim):
    _state_dict = torch.load(fp, map_location=DEVICE)
    if 'nf' in fp:
        model = NF_Model(shape=784)
    else:
        model = GAN_Generator(latent_dim=latent_dim, device=DEVICE)
    model.load_state_dict(_state_dict)
    return model


def interpolate(model, name, n=10, steps=20):
    imgs = torch.zeros(n * steps, 1, 28, 28)
    a = None
    for i in range(n):
        z, a = model.interpolate(a, steps)
        imgs[i*steps:(i+1)*steps, ...] = z.detach().view(-1, 1, 28, 28)
    save_image(imgs, f"figures/{name}_interpolate.png", nrow=steps, normalize=True)


def main(args):
    model = load_model(args.model, latent_dim=args.latent_dim)

    while True:
        interpolate(model, name=args.model.split('_')[0])
        input()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--model', type=str, required=True)
    ARGS = parser.parse_args()
    main(ARGS)

