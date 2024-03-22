from abc import ABC, abstractmethod
from torch import nn
import torch.nn.functional as F

from layers import mlp_layer, conv_layer, projection


class LinResBlock(nn.Module):
    def __init__(self, intput_shape: int, hidden_shape: int, output_shape: int, dropout_p: float = 0.25):
        """ Original Residual block with 2 fully-connected layers

        Parameters
        ----------
        intput_shape : int
            Number of features in the input
        hidden_shape : int
            Number of features in the hidden layer
        output_shape : int
            Number of features in the output
        """
        super(LinResBlock, self).__init__(intput_shape, hidden_shape, output_shape)

        self.fc1 = mlp_layer(intput_shape, hidden_shape, dropout_p=dropout_p)

        # the ReLU is after the concatenation of the skip-connection and the residual function
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_shape, output_shape),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        id_mapping = x
        x = self.fc1(x)
        x = self.fc2(x)
        x += id_mapping
        return F.relu(x)


class ConvResBlock(nn.Module):
    sampling_factor = 1

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        """ Residual block with 2 convolutional layers

        Parameters
        ----------
        input_channels : int
            Number of features in the input
        output_channels : int
            Number of features in the output
        """
        super(ConvResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels)
            # the ReLU is after the concatenation of the skip-connection and the residual function
        )

        self.skip_proj = None  # down / up-sampling if needed for the skip-connection
        if stride != 1 or input_channels != output_channels:
            # if stride != 1, the projection is performed by increasing the stride
            # if stride == 1, the projection is performed by the 1x1 conv itself
            self.skip_proj = projection(input_channels, output_channels, stride)

    def forward(self, x):
        id_mapping = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.skip_proj is not None:
            id_mapping = self.skip_proj(id_mapping)
        x += id_mapping
        return F.relu(x)


class ConvResBlockPre(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        """ Residual block with 2 convolutional layers and a pre-activation

        Parameters
        ----------
        input_channels : int
            Number of features in the input
        output_channels : int
            Number of features in the output
        """
        super(ConvResBlockPre, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        )

        self.skip_proj = None  # down / up-sampling if needed for the skip-connection
        if stride != 1 or input_channels != output_channels:
            # if stride != 1, the projection is performed by increasing the stride
            # if stride == 1, the projection is performed by the 1x1 conv itself
            self.skip_proj = projection(input_channels, output_channels, stride)

    def forward(self, x):
        id_mapping = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.skip_proj is not None:
            id_mapping = self.skip_proj(id_mapping)
        return x + id_mapping


class ConvPlainBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        """ Equivalent of a Residual block with 2 convolutional layers but without the skip-connection. Simply
        performs 2 3x3 convolutions.

        Parameters
        ----------
        input_channels : int
            Number of features in the input
        output_channels : int
            Number of features in the output
        """
        super(ConvPlainBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return F.relu(x)


class BottleNeckBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
