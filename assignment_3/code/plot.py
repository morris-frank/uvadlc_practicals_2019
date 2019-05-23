import matplotlib.pyplot as plt
import torch

plt.style.use('ggplot')


def plot_vae(zdim):
    fp = f"./vae_{zdim}_curves.pt"
    data = torch.load(fp)

    plt.figure(figsize=(3.3, 2), dpi=2*72)
    plt.plot(data['train'], label='train')
    plt.plot(data['val'], label='val')
    plt.xlabel('Epochs')
    plt.ylabel('ELBO')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/vae_{zdim}.pdf")


def plot_nf():
    fp = "./nf_curves.pt"
    data = torch.load(fp)

    plt.figure(figsize=(5, 2), dpi=2*72)
    plt.plot(data['train'], label='train')
    plt.plot(data['val'], label='val')
    plt.xlabel('Epochs')
    plt.ylabel('BPD')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/nf.pdf")


def plot_gan():
    fp = "./gan_curves.pt"
    data = torch.load(fp)

    plt.figure(figsize=(5, 2.2), dpi=2*72)
    plt.plot(data['G'], label='Generator')
    plt.plot(data['D'], label='Discriminator')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/gan.pdf")


if __name__ == '__main__':
    plot_gan()
    plot_vae(2)
    plot_vae(20)
    plot_nf()
