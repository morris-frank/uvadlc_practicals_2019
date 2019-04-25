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


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        self.dim = (batch_size, seq_length, input_dim)
        self.Whx = nn.Parameter(torch.zeros(input_dim, num_hidden))
        self.Whh = nn.Parameter(torch.zeros(num_hidden, num_hidden))
        self.Wph = nn.Parameter(torch.zeros(num_hidden, num_classes))
        for l in [self.Whx, self.Whh, self.Wph]:
            nn.init.kaiming_normal_(l)
        self.bh = nn.Parameter(torch.zeros(num_hidden))
        self.bp = nn.Parameter(torch.zeros(num_classes))
        # h is not a parameter!
        self.activation = nn.Tanh()
        self.device = device
        self.to(device)

    def forward(self, x):
        assert(x.shape == self.dim)
        h = torch.zeros(self.dim[0], self.Whh.shape[0], device=self.device)
        for s in range(self.dim[1]):
            h = self.activation(x[:, s, :] @ self.Whx + h @ self.Whh + self.bh)
        return h @ self.Wph + self.bp
