"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        """
        super(ConvNet, self).__init__()

        def conv(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(True)
            )

        def pool():
            return nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.filter = nn.Sequential(
            conv(n_channels,64),
            pool(),
            conv(64, 128),
            pool(),
            conv(128, 256),
            conv(256, 256),
            pool(),
            conv(256, 512),
            conv(512, 512),
            pool(),
            conv(512, 512),
            conv(512, 512),
            pool(),
            nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        )
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        features = self.filter(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out
