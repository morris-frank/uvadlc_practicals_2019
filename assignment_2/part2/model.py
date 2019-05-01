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

import torch
import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, vocabulary_size, lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0', dropout_prob=0.):
        super(TextGenerationModel, self).__init__()
        self.embed = nn.Embedding(vocabulary_size, vocabulary_size, _weight=torch.eye(vocabulary_size))
        self.embed.weight.requires_grad = False
        self.lstm = nn.LSTM(vocabulary_size, lstm_num_hidden, lstm_num_layers, dropout=dropout_prob)
        self.projection = nn.Linear(lstm_num_hidden, vocabulary_size)
        self.to(device)

    def forward(self, x, h_and_c=None):
        embedding = self.embed(x)
        hidden_states, (h, c) = self.lstm(embedding, h_and_c)
        return self.projection(hidden_states), (h, c)


