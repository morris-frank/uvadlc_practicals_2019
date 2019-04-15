"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
        """
        self.init_std = 0.0001
        self.params = {
            'weight': self.init_std * np.random.randn(in_features, out_features),
            'bias': np.zeros((out_features, 1))
        }
        self.grads = {
            'weight': np.zeros((in_features, out_features)),
            'bias': np.zeros((in_features, 1))
        }

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """
        out = np.add(self.params['weight'] @ x, self.params['bias'])
        self.x = x
        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """
        dx = np.multiply(self.x, dout)
        return dx

class ReLUModule(object):
    """
    ReLU activation module.
    """
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """
        self.grad = x > 0
        out = np.max(x, 0)
        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
        """
        dx = np.multiply(self.grad, dout)
        return dx

class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
        """
        normed_x = np.exp(np.subtract(x, np.max(x, axis=1)[...,np.newaxis]))
        out = np.divide(normed_x, normed_x.sum(axis=1)[...,np.newaxis])
        self.sigma = out
        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
        """
        dx = - self.sigma[:,:,None] * self.sigma[:,None,:]
        dx += np.apply_along_axis(np.diag, 1, self.sigma)
        return dx * dout

class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    def forward(self, x, y):
        """
        Forward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
        """
        out = -np.sum(y * np.log(x))
        return out

    def backward(self, x, y):
        """
        Backward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
        Implement backward pass of the module.
        """
        dx = - np.divide(x, y)
        return dx
