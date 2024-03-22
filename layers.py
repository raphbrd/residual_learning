from torch import nn


def projection(in_channels, out_channels, stride=1):
    """ Perform a 1x1 convolution to project the input to the same shape as the output of the residual block.

    It can either be performed by having a stride > 1 or by using the 1x1 convolutional layer on out_channels >
    in_channels
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
        nn.BatchNorm2d(out_channels)
    )


def conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0, pooling_size=2):
    """ A standard convolution block with a max-pooling layer followed by a ReLU activation

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    stride : int
    padding : int
    pooling_size : int
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.MaxPool2d(pooling_size, pooling_size),
        nn.ReLU()
    )


def linear_layer(in_features, out_features):
    """ A standard fully-connected linear block: pre-batch normalization -> linear layer -> ReLU activation

    Parameters
    ----------
    in_features : int
    out_features : int
    """
    return nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, out_features),
        nn.ReLU()
    )


def mlp_layer(in_features, out_features, dropout_p=0.25):
    """ A standard multi-perceptron linear block with dropout and ReLU activation

    Returns a torch.nn.Sequential module performing : Linear fully-connected layer --> ReLU activation --> Dropout

    Parameters
    ----------
    in_features : int
    out_features : int
    dropout_p : float
        Dropout probability during training. Default is 0.25
    """
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.Dropout(p=dropout_p)
    )
