""" Simple widgets to display on the homepage of the app. """
import os
import streamlit as st
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from data import get_normalized_attr, _load_torch_data
from models import PlainNet, LeNet5, ResNet, DeepMLP
from residuals import ConvResBlock, ConvResBlockPre
from utils import UnNormalize
from viz import plot_sample_image


@st.cache_data
def load_data_st(data_path, dataset_name="MNIST", batch_size=128, val_size=0.15):
    """ Convenient wrapper of the `_load_torch_data` function to cache the data loaders in a streamlit app."""
    return _load_torch_data(data_path, dataset_name, batch_size, val_size)


def display_col_sliders(st_obj, *args):
    """ Display sliders side by side using streamlit columns.

    Parameters
    ----------
    st_obj : streamlit.delta_generator.DeltaGenerator
        The streamlit object to use to display the sliders
    args : tuple
        A tuple of arguments to pass to a streamlit slider
    """
    cols = st_obj.columns(len(args))
    outputs = [None] * len(args)
    for idx, (col, slider) in enumerate(zip(cols, args)):
        outputs[idx] = col.slider(*slider)

    return outputs


def set_dataset_sidebar(st_obj):
    with st_obj.sidebar:
        st_obj.header("Dataset parameters")
        # TODO: use form to get the parameters at once without re-running the app in the mean time
        dataset_name = st.selectbox("Dataset", ["CIFAR10", "MNIST", "SVHN"], index=0)
        batch_size = st.slider("Batch size", 32, 512, 128, step=32)
        validation_size = st.slider("Validation set size (if 0, no validation performed)", 0., .25, .10, step=.05)

    return dataset_name, batch_size, validation_size


def display_datasets_metrics(st_obj, train_loader, val_loader, test_loader, classes):
    col_metrics = st_obj.columns(4)
    col_metrics[0].metric("Training set size", len(train_loader.dataset))
    col_metrics[1].metric("Validation set size", len(val_loader.dataset) if val_loader is not None else 0)
    col_metrics[2].metric("Test set size", len(test_loader.dataset))
    col_metrics[3].metric("Number of classes", len(classes))


def display_sample_img(train_loader, classes, dataset_name):
    # picking a random sample of images from the training set
    images, labels = next(iter(train_loader))

    n_img = 12
    images, labels = images[:n_img], labels[:n_img]
    labels = [classes[label] for label in labels]

    fig = plot_sample_image(images, labels, n_rows=2, n_cols=6, figsize=(9, 3),
                            unnorm=UnNormalize(get_normalized_attr(f"{dataset_name}_MEAN"),
                                               get_normalized_attr(f"{dataset_name}_STD")))
    st.pyplot(fig)


def select_model(input_size, n_classes, models=None):
    if models is None:
        models = ["LeNet5", "MLP", "ResNet", "PlainNet"]
    model_name = st.selectbox("Model", models, index=0)

    # default model
    model = LeNet5(input_size[1], n_classes, image_width=input_size[2])
    if model_name == "MLP":
        model = _create_DeepMLP(input_size, n_classes)
    elif model_name == "ResNet":
        model = _create_ResNet(input_size[1], n_classes)
    elif model_name == "PlainNet":
        model = _create_PlainNet(input_size[1], n_classes)

    return model


def select_optimizer(model):
    optimizer_name = st.selectbox("Optimizer", ["SGD", "Adam"],
                                  format_func=lambda lab: dict(SGD="SGD with momentum", Adam="Adam")[lab], index=0,
                                  disabled=False, help="Most papers uses SGD.")
    learning_rate = st.select_slider("Learning rate (LR)", options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                                     value=0.01)
    columns = st.columns(2, gap="medium")
    # st.info("If either of these LR decay parameters is set to 0, no decay will be performed.")
    scheduler_lr_step = columns[0].slider("LR decay every $n$ epoch", 0, 60, value=40, step=5, help="0 means no decay")
    scheduler_lr_gamma = columns[1].select_slider("LR decay scale", [0, 0.0001, 0.001, 0.01, 0.1], value=0.1,
                                                  help="0 means no decay")

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        st.info("If using Adam, it is recommended to use a smaller learning rate than with SGD and to not set "
                "any learning rate decay.")
    scheduler = None
    if scheduler_lr_gamma > 0 and scheduler_lr_step > 0:
        scheduler = StepLR(optimizer, step_size=scheduler_lr_step, gamma=scheduler_lr_gamma)

    n_epochs = st.slider("Number of epochs", 1, 75, value=15)

    return optimizer, scheduler, n_epochs


def _create_ResNet(input_chs, n_classes):
    st.info("A module is a set of residual blocks with identical hyperparameters. "
            "Each block performs two 3x3 convolutions with the specified output channels. "
            "When required, a projection is automatically added for the identity mapping. "
            "See section 2 of the report for details.")
    n_res_modules = st.slider("Number of modules", 1, 3, 2)
    block_type = st.selectbox("Block type", ["ConvResBlock", "ConvResBlockPre"], index=0,
                              key=f"block_type",
                              help="ConvResBlockPre performs ReLU and BatchNorm before the convolution.")
    shapes = [tuple()] * n_res_modules
    features_shapes = [16] * n_res_modules
    tabs = st.tabs([f"Module {i + 1}" for i in range(n_res_modules)])
    for idx, tab in enumerate(tabs):
        with tab:
            subcol1, subcol2 = st.columns(2, gap="medium")
            n_blocks = subcol1.slider(f"Number of residual blocks", 1, 6, 2, key=f"n_blocks_{idx}")
            out_channels = subcol2.slider(f"Output channels for each 3x3conv", 6, 128, 16 * (idx + 1), step=2,
                                          key=f"n_chs_in_{idx}")
            shapes[idx] = n_blocks
            features_shapes[idx] = out_channels
    block_type = ConvResBlock if block_type == "ConvResBlock" else ConvResBlockPre
    model = ResNet(input_chs, n_classes, module_list=shapes, features_shapes=features_shapes, block_type=block_type)

    return model


def _create_PlainNet(input_chs, n_classes):
    st.info("A module is a set of blocks with identical hyperparameters. "
            "Each block performs two 3x3 convolutions with the specified output channels. This way of building"
            "a network is identical to the ResNets, minus the residual connection."
            "When required, a projection is automatically added for the identity mapping. "
            "See section 2 of the report for details.")
    n_res_modules = st.slider("Number of modules", 1, 3, 2)
    shapes = [tuple()] * n_res_modules
    features_shapes = [16] * n_res_modules
    tabs = st.tabs([f"Module {i + 1}" for i in range(n_res_modules)])
    for idx, tab in enumerate(tabs):
        with tab:
            subcol1, subcol2 = st.columns(2, gap="medium")
            n_blocks = subcol1.slider(f"Number of blocks", 1, 6, 2, key=f"n_blocks_{idx}")
            out_channels = subcol2.slider(f"Output channels for each 3x3conv", 6, 128, 16 * (idx + 1), step=2,
                                          key=f"n_chs_in_{idx}")
            shapes[idx] = n_blocks
            features_shapes[idx] = out_channels
    model = PlainNet(input_chs, n_classes, module_list=shapes, features_shapes=features_shapes)

    return model


def _create_DeepMLP(input_size, n_classes):
    choices = [64, 128, 256, 512, 1024, 2048] * 3
    layers = st.multiselect("Number of units for each hidden layer", choices, default=[128, 64])
    model = DeepMLP(layers=[input_size[2] ** 2 * input_size[1]] + layers + [n_classes])

    return model
