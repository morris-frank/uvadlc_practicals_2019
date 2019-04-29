# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

################################################################################


def greedy_sampling(out, temp):
    return out.argmax()


def temperature_sampling(out, temp):
    dist = torch.softmax(out/temp, dim=0)
    return torch.multinomial(dist, 1).item()


def seq_sampling(model, dataset, seq_length, device, sampler=temperature_sampling, temp=None):
    ramblings = torch.randint(dataset.vocab_size, (1, seq_length), device=device)

    h_and_c = None
    for i in range(1, seq_length):
        out, h_and_c = model.forward(ramblings, h_and_c)
        ramblings[0, i] = sampler(out[0,i, ...].squeeze(), temp)
    text = dataset.convert_to_string(ramblings.numpy().squeeze())
    log = "{};{};{};{}\n".format(time.time(), sampler.__name__, temp, text)
    print(log)
    return log


def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use

    dataset = TextDataset(config.txt_file, config.seq_length)

    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size,
                                config.lstm_num_hidden, config.lstm_num_layers,
                                config.device, 1. - config.dropout_keep_prob)

    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.learning_rate_step, gamma=1.-config.learning_rate_decay)
    accuracies = [0, 1]
    losses = [0, 1]

    step = 0
    while step < config.train_steps:
        for _, (batch_inputs, batch_targets) in enumerate(data_loader):
            step += 1
            # Only for time measurement of step through network
            t1 = time.time()

            device_inputs = torch.stack(batch_inputs, dim=0).to(device)
            device_targets = torch.stack(batch_targets, dim=1).to(device)

            out, _ = model.forward(device_inputs)
            outt = out.transpose(0, 1).transpose(1, 2)
            optimizer.zero_grad()
            loss = criterion.forward(outt, device_targets)
            losses.append(loss.item())
            accuracy = (outt.argmax(dim=1) == device_targets).float().mean()
            accuracies.append(accuracy)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}, LR = {}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        int(config.train_steps), config.batch_size, examples_per_second,
                        accuracies[-1], losses[-1], optimizer.param_groups[-1]['lr']
                ))

            if step % config.sample_every == 0:
                torch.save(model, config.txt_file + '.model')
                log = []
                with torch.no_grad():
                    log.append(seq_sampling(model, dataset, config.seq_length, device, greedy_sampling))
                    for T in [0.5, 1.0, 2.0]:
                        log.append(seq_sampling(model, dataset, config.seq_length, device, temperature_sampling, temp=T))
                with open(config.txt_file + '.generated', 'a') as fp:
                    fp.writelines(log)

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

    print('Done training.')

################################################################################
################################################################################


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=500, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)
