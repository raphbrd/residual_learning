""" Module to load and preprocess data for deep learning tasks in a streamlit app, using PyTorch.

Currently, it aims at loading the following datasets from torchvision:
- MNIST
- CIFAR-10
- SVHN

The main function of this module caches the PyTorch data loaders. It also provides the mean and standard deviation
values of the datasets, which are dynamically accessible using the `__getattr__` method.
"""
import os.path
import numpy as np
from pathlib import Path
from time import ctime

import pandas as pd
import torch.utils.data
import torchvision
from torchvision.transforms import v2
from torch.utils.data import random_split

# distribution parameters on the training set
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)  # 3 channels
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def compute_dataset_mean_std(loader):
    """ Compute the mean and standard deviation of a dataset using the data loader for
    efficiency.

    @see
    """
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    return mean, std


# TODO : improve the hard-coded part of this module
class DatasetFactory:
    """ Factory class to create PyTorch datasets from the dataset name with the appropriate normalization and
    transformation.

    This class should not be instantiated. Use the `create` method to get the dataset object.

    Usage
    -----
    >>> import data
    >>> dataset_name = "MNIST"
    >>> data.DatasetFactory.create(dataset_name, path="/path/to/data")
    """

    datasets = ["MNIST", "CIFAR10", "SVHN"]

    transforms = {
        "MNIST_train": v2.Compose([
            v2.ToTensor(),
            v2.Normalize(MNIST_MEAN, MNIST_STD),
        ]),
        "MNIST_test": v2.Compose([
            v2.ToTensor(),
            v2.Normalize(MNIST_MEAN, MNIST_STD),
        ]),
        "CIFAR10_train": v2.Compose([
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ]),
        "CIFAR10_test": v2.Compose([
            v2.ToTensor(),
            v2.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ]),
        "SVHN_train": v2.Compose([
            v2.ToTensor(),
        ]),
        "SVHN_test": v2.Compose([
            v2.ToTensor(),
        ])
    }

    def __new__(cls):
        raise ValueError("This class should not be instantiated. "
                         "Usage: `DatasetFactory.create(dataset_name, path, **kwargs)`")

    @classmethod
    def create(cls, dataset_name, path, **kwargs):
        """ Create a PyTorch dataset from the dataset name.

        Applies the appropriate normalization and transformation to the dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load. Currently, only "MNIST", "CIFAR10" and "SVHN" are supported
        path : str
            Path to the data directory
        **kwargs : dict
            Additional arguments to pass to the torchvision.datasets.DATASET-NAME class.
            For instance, `train=True` to load the training set or `train=False` to load the test set.
        """
        if dataset_name not in cls.datasets:
            raise ValueError(f"Unknown dataset name: {dataset_name}. To add a new dataset, add the NAME "
                             f"to `DatasetFactory.datasets` list and a corresponding static method `_get_NAME_data`.")

        # if no train argument specified, assuming working with the training set
        transform_name = f"{dataset_name}_train"
        if "train" in kwargs:
            transform_name = f"{dataset_name}_train" if kwargs["train"] else f"{dataset_name}_test"
        transform = cls.transforms[transform_name]

        # SVHN has a different argument name for the train/test split
        if dataset_name == "SVHN" and "train" in kwargs:
            kwargs["split"] = "train" if kwargs["train"] else "test"
            del kwargs["train"]

        return getattr(cls, f"_get_{dataset_name}_data")(path, transform, **kwargs)

    @staticmethod
    def _get_MNIST_data(path, transform, train=True, download=True):
        return torchvision.datasets.MNIST(root=path, train=train, download=download, transform=transform)

    @staticmethod
    def _get_CIFAR10_data(path, transform, train=True, download=True):
        return torchvision.datasets.CIFAR10(root=path, train=train, download=download, transform=transform)

    @staticmethod
    def _get_SVHN_data(path, transform, split="train", download=True):
        return torchvision.datasets.SVHN(root=path, split=split, download=download, transform=transform)


def _load_torch_data(data_path, dataset_name="MNIST", batch_size=128, val_size=0.15):
    """ Get PyTorch data loaders for the specified dataset. The data is normalized and transformed to tensors.

    If specified, the validation set is created from the training set with an 85/15 random split using a fixed seed.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load. Currently, only "MNIST" and CIFAR10 are supported
    batch_size : int
    val_size : float
        Proportion of the training set to use for the validation set. Default is 0.1. If 0,
        no validation set is created.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        Training data loader
    validation_loader : torch.utils.data.DataLoader | None
        Validation data loader or None if `use_validation` is False
    test_loader : torch.utils.data.DataLoader
        Test data loader
    classes : tuple
        a tuple of strings with the class names for the classification task
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist.")

    generator = torch.Generator().manual_seed(42)

    # train_set = getters[dataset_name](root=data_path, train=True, download=True, transform=transform)
    train_set = DatasetFactory.create(dataset_name, path=data_path, train=True, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = None

    if val_size > 0:
        val_length = int(len(train_set) * val_size)
        train_length = len(train_set) - val_length

        train_set, validation_set = random_split(train_set, [train_length, val_length], generator=generator)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False,
                                                        num_workers=0)

    # test_set = getters[dataset_name](root=data_path, train=False, download=True, transform=transform)
    test_set = DatasetFactory.create(dataset_name, path=data_path, train=False, download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    classes = test_set.classes if hasattr(test_set, "classes") else np.unique(test_set.labels)

    return train_loader, validation_loader, test_loader, classes


def get_classes_labels(test_set):
    """ Get the class labels from a PyTorch dataset object. """
    # not all torchvision datasets have a `classes` attribute
    return test_set.classes if hasattr(test_set, "classes") else np.unique(test_set.labels)


def file_list_in_directory(path):
    p = Path(path)
    files = []
    for i in p.rglob('*.pth*'):
        files.append((i.name, ctime(i.stat().st_ctime)))

    return pd.DataFrame.from_records(files, columns=["Name", "Creation time"]).sort_values(
        by="Creation time").reset_index(drop=True)


def get_normalized_attr(name):
    """ Return the mean and standard deviation of the specified dataset used in the transforms. Useful
    to un-normalize data for visualization.

    Default to MEAN = (1.0, ) and STD = (0.0, ).

    Examples
    --------
    >>> import data
    >>> dataset_name = "MNIST"
    >>> data.get_normalized_attr(f"{dataset_name}_MEAN")
    (0.1307,)
    """
    if "MEAN" not in name and "STD" not in name:
        raise ValueError(f"Expected a string containing 'MEAN' or 'STD' to retrieve normalization parameters,"
                         f" got {name}.")

    if name in globals():
        return globals()[name]

    # if not found, return default values:
    if "MEAN" in name:
        return (0.0,)
    if "STD" in name:
        return (1.0,)
