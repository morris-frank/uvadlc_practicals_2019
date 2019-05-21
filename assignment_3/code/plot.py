from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torch

plt.style.use('ggplot')


def plot_vae(zdim):
    fp = f"./vae_{zdim}_curves.pt"
    torch.load(fp)
    breakpoint()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nf', action='store_true')

    args = parser.parse_args()

    if args.nf:
        plot_vae(z)
