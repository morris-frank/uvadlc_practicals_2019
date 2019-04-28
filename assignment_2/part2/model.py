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

import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0', dropout_prob=0.):

        super(TextGenerationModel, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.emsize = lstm_num_hidden
        self.embed = nn.Embedding(vocabulary_size, self.emsize)
        self.lstm = nn.LSTM(self.emsize, lstm_num_hidden, lstm_num_layers, dropout=dropout_prob)
        self.projection = nn.Linear(lstm_num_hidden, vocabulary_size)
        self.to(device)

    def forward(self, x, h_and_c=None):
        # assert x.shape == (self.seq_length, self.batch_size)
        embedding = self.embed(x)
        hidden_states, (h, c) = self.lstm(embedding, h_and_c)
        hidden_states.transpose_(0, 1)
        return self.projection(hidden_states), (h, c)


