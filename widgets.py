""" Simple widgets to display on the homepage of the app. """
import os
import streamlit as st

from data import get_normalized_attr, _load_torch_data
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


def select_model(st_obj):
    pass


def get_optimizer(st_obj):
    pass


def _create_ResNet(st_obj):
    pass


def _create_ResMLP(st_obj):
    pass


def _create_DeepMLP(st_obj):
    pass


def _create_LeNet(st_obj):
    pass
