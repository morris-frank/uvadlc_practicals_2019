################################################################################
# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pickle
import time
from datetime import datetime
from itertools import product
import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PalindromeDataset
from lstm import LSTM
from vanilla_rnn import VanillaRNN


RES_FILE = 'palindrome.obj'


def experiment(config, repeat=5, stride=2, start=15):
    config.quite = True
    if os.path.exists(RES_FILE):
        with open(RES_FILE, 'rb') as fp:
            results = pickle.load(fp)
    else:
        results = {'RNN': {}, 'LSTM': {}}
    lengths = range(start, config.input_length, stride)
    for _ in range(repeat):
        for model, length in product(results.keys(), lengths):
            print('{} with lenght {}'.format(model, length))
            config.model_type, config.input_length = model, length
            results[model].setdefault(length, ([], []))
            accs, loss = train(config)
            results[model][length][0].append(accs)
            results[model][length][1].append(loss)
        with open(RES_FILE, 'wb') as fp:
            pickle.dump(results, fp)


def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    tol = 0.

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden,
                           config.num_classes, config.batch_size, device)
    else:
        model = LSTM(config.input_length, config.input_dim, config.num_hidden,
                     config.num_classes, config.batch_size, device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    accuracies = [0, 1]
    losses = [0, 1]

    if config.quite:
        bar = tqdm(total=config.train_steps)
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Only for time measurement of step through network
        t1 = time.time()

        batch_inputs = batch_inputs[..., None]
        batch_inputs.to(device)
        batch_targets.to(device)

        # FORWARD, BACKWARD, AND STEP
        out = model.forward(batch_inputs)
        model.zero_grad()
        loss = criterion(out, batch_targets)
        loss.backward()

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################
        optimizer.step()

        # Add more code here ...
        accuracy = (out.argmax(dim=1) == batch_targets.long()).float().mean()
        losses.append(loss.item())
        accuracies.append(accuracy.item())

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0 and not config.quite:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracies[-1], losses[-1]
            ))
        if config.quite:
            bar.update()
        if step == config.train_steps or np.isclose(losses[-1], losses[-2], tol):
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break
    print('Done training.')
    return accuracies[2:], losses[2:]


################################################################################
################################################################################


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--quite', action='store_true', default=False)
    config = parser.parse_args()

    # Train the model
    if config.run:
        experiment(config)
    else:
        train(config)
