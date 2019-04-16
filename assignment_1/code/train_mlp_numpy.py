"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    """
    accuracy = np.mean((predictions.argmax(axis=1) == targets.argmax(axis=1)))
    return accuracy

def train():
    """
    Performs training and evaluation of MLP model.
    """

    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = [DNN_HIDDEN_UNITS_DEFAULT]

    data = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)

    n_inputs = np.prod(data['train'].images.shape[1:])
    n_classes = data['train'].labels.shape[1]
    n_test = data['test'].images.shape[0]

    X_test,y_test = data['test'].next_batch(n_test)
    X_test = X_test.reshape((n_test, n_inputs))

    net = MLP(n_inputs, dnn_hidden_units, n_classes)
    loss_func = CrossEntropyModule()

    losses = {'train': [], 'test': []}
    accuracies = {'train': [], 'test': []}

    for s in range(FLAGS.max_steps):
        X,y = data['train'].next_batch(FLAGS.batch_size)
        X = X.reshape((FLAGS.batch_size, n_inputs))

        #FORWARD
        out = net.forward(X)

        #BACKWARD
        dLoss = loss_func.backward(out, y)
        net.backward(dLoss)

        #UPDATE
        for grad_layer in filter(lambda x:hasattr(x, 'grads'), net.layers):
            grad_layer.params['weight'] -= FLAGS.learning_rate * grad_layer.grads['weight']
            grad_layer.params['bias'] -= FLAGS.learning_rate * grad_layer.grads['bias']


        #Evaluation
        if s%FLAGS.eval_freq == 0 or s == FLAGS.max_steps-1:
            losses['train'].append(loss_func.forward(out, y))
            accuracies['train'].append(accuracy(out, y))

            out = net.forward(X_test)
            losses['test'].append(loss_func.forward(out, y_test))
            accuracies['test'].append(accuracy(out, y_test))
            
            print('Iter {:04d}: Test: {:.2f} ({:f}), Train: {:.2f} ({:f})'.format(
                s, 100*accuracies['test'][-1], losses['test'][-1], 100*accuracies['train'][-1], losses['train'][-1]))


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
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()