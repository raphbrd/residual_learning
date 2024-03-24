""" Models for image classification

@author : Raphael Bordas
"""
import inspect

import torch
from torch import nn
import torch.nn.functional as F

from layers import conv_layer, mlp_layer, linear_layer
from residuals import LinResBlock, ConvResBlock, ConvPlainBlock
from utils import get_conv_out_dim


def set_grad_analysis(model: nn.Module, grads: list) -> list[torch.utils.hooks.RemovableHandle]:
    """ Register a hook to analyze the gradients of the model during the backward pass.

    Only set the hook on the Conv2d layers of the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to analyze
    grads : list
        The list to store the gradients of the model during the backward pass

    Returns
    -------
    list
        A list of handles used to remove the hooks with the `remove_grad_analysis` function
    """

    handles = []

    def hook_wrapper(name):
        def hook_fn(module, grad_input, grad_output):
            grads.append((name, module, grad_input, grad_output))

        return hook_fn

    # name of the layers, useful to identify the layers of the residual blocks
    names = [name for name, _ in model.named_modules()]
    for name, layer in zip(names, model.modules()):
        if isinstance(layer, nn.Conv2d):
            handle = layer.register_full_backward_hook(hook_wrapper(name))
            handles.append(handle)

    return handles


def remove_grad_analysis(handles):
    """ Remove the hooks used to analyze the gradients of the model during the backward pass """
    for handle in handles:
        handle.remove()


def xavier_weights(model):
    """ Initialize the weights of a PyTorch model using the Xavier initialization (normal distribution) """
    for n, p in model.named_parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_normal_(p)


def kaiming_weights(model):
    """ Initialize the weights of a PyTorch model using the Kaiming initialization (normal distribution) """
    for n, p in model.named_parameters():
        if p.dim() > 1:
            torch.nn.init.kaiming_normal_(p)


class DeepMLP(nn.Module):
    def __init__(self, layers: list):
        """ Very deep multi-layer perceptron for classification tasks.

        @see https://arxiv.org/pdf/1003.0358.pdf for an example of a very deep MLP used in MNIST classification.

        Each layer is a fully-connected layer with ReLU activation and dropout (default 0.25).

        Parameters
        ----------
        layers : list
            A list of the number of features in each layer. The first item of the list is the number of features in the
            input layer. The last item is the number of classes to predict. Therefore, the list must have at least 3
            items (e.g., in the case of the simplest one-hidden-layer MLP).

        Examples
        --------
        # Create a 3-layer MLP for MNIST classification (28x28 images)
        >>> mlp = DeepMLP(layers=[28 * 28, 128, 64, 10])  # L1 of 128 units --> L2 of 64 units --> 10 [outputs]
        """
        super().__init__()

        if len(layers) < 3:
            raise ValueError("The number of layers must be at least 3 (i.e., at least one hidden layer)")

        blocks = []
        for shape_in, shape_out in list(zip(layers[:-1], layers[1:]))[:-1]:
            blocks.append(mlp_layer(shape_in, shape_out))
        self.fc = nn.Sequential(*blocks)

        # the last layer does not have dropout nor ReLU
        self.fc_out = nn.Linear(layers[-2], layers[-1])

    def forward(self, x):
        # flatten the input if needed
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)

        # the list of fully-connected layers was defined as a torch.nn.Sequential object
        # and therefore implement the forward method
        x = self.fc(x)

        return F.softmax(self.fc_out(x), dim=1)


class ResMLP(nn.Module):
    def __init__(self, input_shape, output_shape, last_layer_size=64, n_blocks=4,
                 block_shapes=None, **block_kwargs):
        """ Residual multi-layer perceptron """
        super().__init__()

        if block_shapes is None:
            block_shapes = [(128, 256, 128), ] * n_blocks

        if len(block_shapes) != n_blocks:
            raise ValueError(f"Expected n_blocks={n_blocks} shapes for the residual blocks, got {len(block_shapes)}.")

        # first layer is simply a mapping from input to the first hidden layer size
        self.fc1 = mlp_layer(input_shape, block_shapes[0][0])

        self.res_blocks = nn.ModuleList([
            LinResBlock(*shape, **block_kwargs) for shape in block_shapes
        ])
        self.fc2 = mlp_layer(block_shapes[-1][-1], last_layer_size)
        self.fc3 = nn.Linear(last_layer_size, output_shape)

    def forward(self, x):
        # flatten the input if needed
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.fc2(x)

        return F.softmax(self.fc3(x), dim=1)


class ResNet(nn.Module):
    """ Main class to implement a Residual Network in PyTorch for image classification """

    def __init__(self, input_channels, n_classes, block_type=ConvResBlock, module_list=None,
                 features_shapes=None):
        """ Initialize the ResNet model

        Parameters
        ----------
        input_channels : int
            Number of channels of the input images
        n_classes : int
            Number of classes to predict
        block_type : class
            The class of the residual block to use. It must be a subclass of torch.nn.Module and implement a forward
            method. Default is ConvResBlock.
        module_list : list
            Number of residual blocks for each module of the network. It must be a list of the same length as
            `features_shapes` and the number of module. A module is a sequence of residual blocks with the same
            number of output channels. If None (default) will be set to [2, 2, 2].
        features_shapes : list
            Number of output channels the layers of each module of the network. If None (default) will be set
            to [16, 32, 64]

        Examples
        --------
        # Create a ResNet with 1 residual module consisting of 1 residual block (= 2 conv 3x3) with 16 output channels,
        1 residual module with 3 residual blocks with 32 output channels and 1 residual module with 2 residual blocks
        with 64 output channels. The input images have 3 channels and the network must predict 10 classes.
        >>> resnet = ResNet(input_channels=3,n_classes=10,module_list=[1, 3, 2],features_shapes=[16, 32, 64])
        """
        super(ResNet, self).__init__()

        # TODO : update the check of the block type
        self.block_cls = block_type
        if module_list is None:
            module_list = [2, 2, 2]  # 2 layers for each of the 3 blocks
        if features_shapes is None:
            features_shapes = [16, 32, 64]  # out_channels of each residual block

        if len(features_shapes) != len(module_list):
            raise ValueError(
                f"Expected {module_list} features shapes for the residual blocks, got {len(features_shapes)}.")

        # first layer identical to the first layer of a standard CNN
        # output = 32x32xblock_shapes[0][0] feature maps (padding = 1 to keep same feature map size)
        self.conv1 = nn.Conv2d(input_channels, features_shapes[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(features_shapes[0])

        # first module is different as it has the same number of input and output channels
        blocks_list = [self._create_res_block(features_shapes[0], features_shapes[0], module_list[0], stride=1)]

        # shifting the features shapes to get the output channels of the previous block as the input channels
        # of the next one
        in_chs_list = features_shapes[:-1]
        out_chs_list = features_shapes[1:]
        # the following modules have a stride of 2
        for in_chs, out_chs, n_block in zip(in_chs_list, out_chs_list, module_list[1:]):
            blocks_list.append(self._create_res_block(in_chs, out_chs, n_block, stride=2))
        self.res_blocks = nn.ModuleList(blocks_list)

        # AvgPooling layer identical to the original ResNet
        self.pooling = nn.AvgPool2d(8)

        n_convs = sum(module_list) * 2 + 1  # number of 3x3 conv in the whole network

        # counting the number of strides in the residual modules (except the first one that is different)
        # first conv has stride 2 (downsampling the input), then only stride 1
        strides_blocks = []
        for layer in module_list[1:]:
            strides_blocks += [2, 1] + [1, 1] * (layer - 1)

        flat_dim = get_conv_out_dim(
            input_size=32,
            kernel_size=[3] * n_convs + [8],
            padding=[1] * n_convs + [0],
            stride=[1] + [1, 1] * module_list[0] + strides_blocks + [8]
        )

        self.fc3 = nn.Linear(features_shapes[-1] * flat_dim ** 2, n_classes)

    def _create_res_block(self, input_channels, out_channels, n_block, stride=1):
        blocks = []
        # TODO refactor this part to avoid the if statement
        # only the first block of the module has a stride of 2
        if stride > 1:
            blocks.append(self.block_cls(input_channels, out_channels, stride=2))
        else:
            blocks.append(self.block_cls(input_channels, out_channels, stride=1))

        # the following blocks have a stride of 1
        for idx in range(1, n_block):
            blocks.append(self.block_cls(out_channels, out_channels, stride=1))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        for idx, res_block in enumerate(self.res_blocks):
            x = res_block(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class PlainNet(ResNet):
    """ Class used to rename a Plain net identical to a ResNet as a Plain Network. In the case, the network will indeed
     be built as a ResNet but with the ConvPlainBlock as the residual block. """

    def __init__(self, *args, **kwargs):
        """ Initialize a PlainNet as a ResNet with ConvPlainBlock as the residual block """
        if "block_type" in kwargs and kwargs["block_type"] != ConvPlainBlock:
            raise ValueError(f"Expected block_type='ConvPlainBlock' for a PlainNet, got {kwargs['block_type']}.")

        # in case the block type is not specified
        kwargs["block_type"] = ConvPlainBlock
        super(PlainNet, self).__init__(*args, **kwargs)  # ResNet constructor is sufficient


class LeNet5(nn.Module):
    def __init__(self, input_channels, output_channels, image_width=32):
        """ LeNet-5 architecture for image classification

        The images are assumed to be square, as the width needs to be specified to compute the number
        of units in the first fully-connected layer.
        The output of the network is a softmax distribution over the classes (i.e. probabilities for each class).

        Parameters
        ----------
        input_channels : int
            Number of channels of the input images
        output_channels : int
            Number of classes to predict
        image_width : int
            Width of the input images
        """
        super().__init__()

        self.conv1 = conv_layer(input_channels, 6, kernel_size=5)
        self.conv2 = conv_layer(6, 16, kernel_size=5)

        self.flat = nn.Flatten()
        h_size_out = get_conv_out_dim(image_width, kernel_size=[5, 2, 5, 2], padding=[0, 0, 0, 0],
                                      stride=[1, 2, 1, 2])

        self.fc1 = linear_layer(h_size_out ** 2 * 16, 120)
        self.fc2 = linear_layer(120, 84)
        # the last layer does not have a ReLU activation
        self.batch3 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, output_channels)

    def forward(self, x):
        # convolutions
        x = self.conv1(x)
        x = self.conv2(x)

        # fully-connected layers
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.batch3(x)
        x = self.fc3(x)

        return F.softmax(x, dim=1)
