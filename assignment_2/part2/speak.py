import torch
from argparse import ArgumentParser
from dataset import TextDataset


def seq_sampling(model, dataset, seq, seq_length, temp=None):
    pivot = torch.Tensor([[seq[0]]]).long()
    ramblings = seq

    h_and_c = None
    for i in range(1, seq_length):
        out, h_and_c = model.forward(pivot, h_and_c)
        if i < len(seq):
            pivot[0, 0] = seq[i]
        else:
            if temp is None or temp == 0:
                pivot[0, 0] = out.squeeze().argmax()
            else:
                dist = torch.softmax(out.squeeze()/temp, dim=0)
                pivot[0, 0] = torch.multinomial(dist, 1)
            ramblings.append(pivot[0, 0].item())
    return dataset.convert_to_string(ramblings)


def speak(mpath, dpath, length):
    model = torch.load(mpath, map_location='cpu')
    dataset = torch.load(dpath)
    while True:
        _input = input('give me a string? ')
        idxs = dataset.convert_from_string(_input.lower())
        text = seq_sampling(model, dataset, idxs, length, temp=0.5)
        print(text)


def main():
    parser = ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--length', type=int, default=200)
    args = parser.parse_args()
    speak(args.path + '.model', args.path + '.dataset', args.length)


if __name__ == '__main__':
    main()
