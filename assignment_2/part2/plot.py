import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pandas as pd
from os.path import splitext


plt.style.use('ggplot')


def plot(fpath):
    df = pd.read_csv(fpath, sep=',', names=['Step', 'batch_size', 'Speed', 'Accuracy', 'Loss', 'LR'])
    df[df.Step < 1e5].plot('Step', ['Accuracy', 'Loss', 'LR'], subplots=True, layout=(1, 3), figsize=(13, 3))
    plt.tight_layout()
    plt.savefig(splitext(fpath)[0] + '.pdf')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    plot(args.file)
