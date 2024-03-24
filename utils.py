import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from prettytable import PrettyTable


class UnNormalize(object):
    """ Un-normalize a tensor image. Useful to visualize the original image."""

    def __init__(self, mean: tuple, std: tuple):
        """ Initialize the UnNormalize object with a tuple of mean and standard deviation values.

        The length of each tuple must be equal to the number of channels in the image to un-normalize.

        Parameters
        ----------
        mean : tuple
            The mean value for each channel
        std : tuple
            The standard deviation for each channel
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tensor : torch.Tensor
            Image of size (C, H, W) to be un-normalized.

        Returns
        -------
        torch.Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def get_device():
    """ Get the device (CPU or GPU) that is available """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_conv_out_dim(input_size, kernel_size, padding, stride):
    """ Compute the dimension of a convolutional layer, assuming the image is squared

    If several layers, the parameters must be lists of the same length. The code assumes that the
    parameters are ordered from the first layer to the last.

    Parameters
    ----------
    input_size : int
        Width (or height) of the square image
    kernel_size : int or list
    padding : int or list
    stride : int or list

    Returns
    -------
    The output width of the convolutional layer

    Examples
    --------
    >>> get_conv_out_dim(32, 3, 1, 1) # output size of a 3x3 convolution with padding of 1 and stride of 1
    32
    >>> get_conv_out_dim(32, [5, 3], [1, 0], [1, 1]) # 5x5 convolution followed by 3x3 convolution
    28
    """
    if isinstance(kernel_size, tuple) or isinstance(kernel_size, list):

        if len(kernel_size) != len(padding) or len(kernel_size) != len(stride) or len(kernel_size) < 1:
            raise ValueError("kernel_size, padding and stride must have the same length")

        for k, p, s in zip(kernel_size, padding, stride):
            if s < 1:
                raise ValueError("stride must be greater than 0")
            # recursive call for sake of simplicity
            input_size = get_conv_out_dim(input_size, k, p, s)

        return input_size

    # at this point stride should be a list
    if stride < 1:
        raise ValueError("stride must be greater than 0")

    return int((input_size - kernel_size + 2 * padding) / stride + 1)


def get_input_size(loader):
    """ Get the size of the input images from a torchvision DataLoader """
    images, _ = next(iter(loader))
    input_size = images.shape

    return input_size


def count_parameters(model, verbose=True):
    """ Count the number of parameters in a model

    @see: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model

    Parameters
    ----------
    model : torch.nn.Module
        The model to count the parameters from
    verbose : bool
        Whether to print the parameters or not

    Returns
    -------
    PrettyTable:
        A table with the model parameters
    int:
        The total number of parameters
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if verbose:
        print(table)
        print(f"Total Trainable Params: {total_params}")
    return table, total_params
