import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

plt.style.use('ggplot')


def smooth(x, win=5):
    k = np.ones(win, 'd')/win
    s = np.r_[x[win-1:0:-1], x, x[-2:-win-1:-1]]
    return np.convolve(k, s, mode='valid')[:len(x)]


def create_padded_array(llist):
    w, h = max([len(x) for x in llist]), len(llist)
    arr = np.full((h,w), np.nan)
    for idx, _el in enumerate(llist):
        arr[idx, 0:len(_el)] = _el
        arr[idx, len(_el):] = _el[-1]
    return arr


def plot():
    xlim = 1500
    co = 2
    with open('palindrome.obj', 'rb') as fp:
        results = pickle.load(fp)

    cms = {'RNN': cm.get_cmap('Greens', len(results['RNN'])+co),
           'LSTM': cm.get_cmap('Oranges', len(results['LSTM'])+co)}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    for model, cmap in cms.items():
        for idx, (length, (accs, losss)) in enumerate(results[model].items()):
            acc = create_padded_array(accs).mean(axis=0)[:xlim]
            loss = create_padded_array(losss).mean(axis=0)[:xlim]
            aline, = ax1.plot(range(len(acc)), smooth(acc), color=cmap(idx+co), alpha=0.9)
            lline, = ax2.plot(range(len(loss)), smooth(loss), color=cmap(idx+co), alpha=0.9)
        for line in [aline, lline]:
            line.set_label(model)
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Loss')
    ax2.set_ylim(0, 2.5)
    for ax in [ax1, ax2]:
        ax.set_xlabel('training steps')
        ax.set_xlim(0, xlim)
        ax.legend()
    plt.tight_layout()
    plt.savefig('palindrome.pdf')


if __name__ == '__main__':
    plot()
