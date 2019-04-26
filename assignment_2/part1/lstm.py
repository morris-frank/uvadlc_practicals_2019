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

import torch
import torch.nn as nn

################################################################################


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        self.batch_size, self.seq_length = batch_size, seq_length
        self.input_dim, self.num_hidden = input_dim, num_hidden

        # Building weights and biases
        self.W, self.b = nn.ParameterDict(), nn.ParameterDict()
        self.activation, self.gate = {}, {}
        for i in ['g', 'i', 'f', 'o']:
            self.W[i + 'x'] = nn.Parameter(torch.zeros(input_dim, num_hidden))
            self.W[i + 'h'] = nn.Parameter(torch.zeros(num_hidden, num_hidden))
            self.b[i] = nn.Parameter(torch.zeros(num_hidden))
            self.activation[i] = nn.Sigmoid()
        self.W['ph'] = nn.Parameter(torch.zeros(num_hidden, num_classes))
        self.b['p'] = nn.Parameter(torch.zeros(num_classes))
        self.activation['g'] = nn.Tanh()

        # Initalize weights
        for l in self.W.values():
            nn.init.kaiming_normal_(l)

        self.device = device
        self.to(device)

    def forward(self, x):
        assert x.shape == (self.batch_size, self.seq_length, self.input_dim)
        h = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        c = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        for s in range(self.seq_length):
            for i in ['g', 'i', 'f', 'o']:
                self.gate[i] = self.activation[i](x[:, s, :] @ self.W[i + 'x'] + h @ self.W[i + 'h'] + self.b[i])
            c = self.gate['g'] * self.gate['i'] + c * self.gate['f']
            h = torch.tanh(c) * self.gate['o']
        return h @ self.W['ph'] + self.b['p']
