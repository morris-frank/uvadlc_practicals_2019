import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

#plt.style.use('seaborn-paper')
plt.style.use('ggplot')


def smooth(x, win=18):
    k = np.ones(win, 'd')/win
    s = np.r_[x[win-1:0:-1], x, x[-2:-win-1:-1]]
    return np.convolve(k, s, mode='valid')[:len(x)]


def plot():
    with open('palindrome.obj', 'rb') as fp:
        results = pickle.load(fp)

    cms = {'RNN': cm.get_cmap('Blues', len(results['RNN'])),
           'LSTM': cm.get_cmap('Oranges', len(results['LSTM']))}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    for model, cmap in cms.items():
        for idx, (length, (acc, loss)) in enumerate(results[model].items()):
            aline, = ax1.plot(range(len(acc)), smooth(acc), color=cmap(idx), alpha=0.9)
            lline, = ax2.plot(range(len(loss)), smooth(loss), color=cmap(idx), alpha=0.9)
        for line in [aline, lline]:
            line.set_label(model)
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Loss')
    ax2.set_ylim(0, 3)
    for ax in [ax1, ax2]:
        ax.set_xlabel('training steps')
        ax.set_xlim(0, 1500)
        ax.legend()
    plt.tight_layout()
    plt.savefig('palindrome.pdf')


if __name__ == '__main__':
    plot()
