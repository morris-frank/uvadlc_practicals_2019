"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
from convnet_pytorch import ConvNet
import cifar10_utils
import matplotlib.pyplot as plt
import sys

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
OUTPUT_DIR = '~/'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      targets: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    """
    return np.mean((predictions.cpu().detach().data.numpy().argmax(axis=1) == targets.cpu().data.numpy()))


def train():
    """
    Performs training and evaluation of ConvNet model.
    """

    # DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    data = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_classes = data['train'].labels.shape[1]
    n_test = data['test'].images.shape[0]

    data_test = data['test']

    net = ConvNet(3, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=FLAGS.learning_rate)

    losses = {'train': [], 'test': []}
    accuracies = {'train': [], 'test': []}
    eval_steps = []

    for s in range(FLAGS.max_steps):
        x, y = data['train'].next_batch(FLAGS.batch_size)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y.argmax(axis=1)).long().to(device)

        # FORWARD, BACKWARD, AND STEP
        out = net.forward(x)
        net.zero_grad()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        # Evaluation
        if s % FLAGS.eval_freq == 0 or s == FLAGS.max_steps - 1:
            eval_steps.append(s)
            losses['train'].append(loss)
            accuracies['train'].append(accuracy(out, y))

            test_losses = []
            test_accuracies = []
            for _ in range(0, n_test, FLAGS.batch_size):
                x_test, y_test = data_test.next_batch(FLAGS.batch_size)
                x_test = torch.from_numpy(x_test).to(device)
                y_test = torch.from_numpy(y_test.argmax(axis=1)).long().to(device)
                out = net.forward(x_test)
                test_losses.append(criterion(out, y_test).item())
                test_accuracies.append(accuracy(out, y_test))
            losses['test'].append(np.mean(test_losses))
            accuracies['test'].append(np.mean(test_accuracies))

            print('Iter {:04d}: Test: {:.2f} ({:f}), Train: {:.2f} ({:f})'.format(
                s, 100 * accuracies['test'][-1], losses['test'][-1],
                100 * accuracies['train'][-1], losses['train'][-1]))

    # Plotting
    # for d, n in [(accuracies, 'Accuracy'), (losses, 'Loss')]:
    #     plt.figure()
    #     plt.plot(eval_steps, d['train'], label='train')
    #     plt.plot(eval_steps, d['test'], label='test')
    #     plt.xlabel('Step')
    #     plt.ylabel(n)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig('conv_' + n.lower() + '.pdf')

    print('Best testing loss: {:.2f} accuracy: {:.2f}'.format(np.min(losses['test']), 100*np.max(accuracies['test'])))


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    open('torch.log', 'w').close()
    sys.stdout = open('torch.log', 'w')

    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    main()
