import torch
import torch.nn as nn
import torch.autograd

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################


class CustomBatchNormAutograd(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    The operations called in self.forward track the history if the input tensors have the
    flag requires_grad set to True. The backward pass does not need to be implemented, it
    is dealt with by the automatic differentiation provided by PyTorch.
    """

    def __init__(self, n_neurons, ε=1e-5):
        """
        Initializes CustomBatchNormAutograd object.

        Args:
          n_neurons: int specifying the number of neurons
          ε: small float to be added to the variance for stability
        """
        super(CustomBatchNormAutograd, self).__init__()
        self.ε = ε
        self.β = nn.Parameter(torch.zeros(n_neurons))
        self.γ = nn.Parameter(torch.ones(n_neurons))
        self.n_neurons = n_neurons

    def forward(self, inputs: torch.Tensor):
        """
        Compute the batch normalization

        Args:
          inputs: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor
        """
        assert inputs.size(1) == self.n_neurons
        μ = inputs.mean(0)
        σ = torch.sqrt(inputs.var(0, unbiased=False) + self.ε)
        x = (inputs - μ) / σ

        out = self.γ * x + self.β
        return out


######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
    """
    This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
    Using torch.autograd.Function allows you to write a custom backward function.
    The function will be called from the nn.Module CustomBatchNormManualModule
    Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
    pass is done via the backward method.
    The forward pass is not called directly but via the apply() method. This makes sure that the context objects
    are dealt with correctly. Example:
      my_bn_fct = CustomBatchNormManualFunction()
      normalized = fct.apply(input, gamma, beta, eps)
    """

    @staticmethod
    def forward(ctx, input, gamma, beta, eps=1e-5):
        """
        Compute the batch normalization

        Args:
          ctx: context object handling storing and retrival of tensors and constants and specifying
               whether tensors need gradients in backward pass
          input: input tensor of shape (n_batch, n_neurons)
          gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
          beta: mean bias tensor, applied per neuron, shpae (n_neurons)
          eps: small float added to the variance for stability
        Returns:
          out: batch-normalized tensor
        """
        assert input.size(1) == gamma.size(0)

        μ = input.mean(0)
        inputμ = input - μ
        σ = input.var(0, unbiased=False)
        invσ = torch.reciprocal(torch.sqrt(σ + eps))
        input_hat = inputμ * invσ
        out = gamma * input_hat + beta

        ctx.save_for_backward(inputμ, σ, invσ, input_hat, gamma)
        ctx.eps = eps

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute backward pass of the batch normalization.

        Args:
          ctx: context object handling storing and retrival of tensors and constants and specifying
               whether tensors need gradients in backward pass
          grad_output: The previous grad output
        Returns:
          out: tuple containing gradients for all input arguments
        """

        N, D = grad_output.shape

        if ctx.needs_input_grad[2]:
            grad_beta = grad_output.sum(0)
        else:
            grad_beta = None

        inputμ, σ, invσ, input_hat, gamma = ctx.saved_tensors

        grad_gamma = None
        grad_input = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_gamma_x = grad_output
            grad_gamma = (grad_gamma_x*input_hat).sum(0)

        if ctx.needs_input_grad[0]:
            grad_xhat = grad_gamma_x * gamma
            grad_ivar = (grad_xhat*inputμ).sum(0)

            grad_sqrt_var = -invσ**2 * grad_ivar

            grad_var = 0.5 * 1. / torch.sqrt(σ+ctx.eps) * grad_sqrt_var

            grad_sq = torch.ones(N,D, dtype=grad_var.dtype) * (1/N) * grad_var
            grad_inputμ = grad_xhat * invσ + 2*inputμ*grad_sq
            grad_μ = -1 * grad_inputμ.sum(0)

            grad_x2 = torch.ones(N,D, dtype=grad_μ.dtype) * (1/N) * grad_μ
            grad_input = grad_inputμ + grad_x2

        if not ctx.needs_input_grad[1]:
            grad_gamma = None
        return grad_input, grad_gamma, grad_beta, None


######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    In self.forward the functional version CustomBatchNormManualFunction.forward is called.
    The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
    """

    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomBatchNormManualModule object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability
        """
        super(CustomBatchNormManualModule, self).__init__()

        self.ε = eps
        self.β = nn.Parameter(torch.zeros(n_neurons))
        self.γ = nn.Parameter(torch.ones(n_neurons))
        self.n_neurons = n_neurons

    def forward(self, inputs):
        """
        Compute the batch normalization via CustomBatchNormManualFunction

        Args:
          inputs: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor
        """
        assert inputs.size(1) == self.n_neurons
        bn_func = CustomBatchNormManualFunction()
        out = bn_func.apply(inputs, self.γ, self.β, self.ε)
        return out
