import matplotlib.pyplot as plt
import seaborn as sns
import torch

from utils import UnNormalize


def plot_sample_image(images, labels, n_rows=3, n_cols=8, unnorm=UnNormalize((0.1307,), (0.3081,)),
                      figsize=(9, 4)):
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

    if n_rows * n_cols > 1:
        axes = axes.flatten()

    for i in range(n_rows * n_cols):
        item = images[i]
        axes[i].imshow((unnorm(item) * 255).to(torch.uint8).permute((1, 2, 0)), cmap="gray")
        axes[i].set_title(labels[i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    fig.tight_layout()

    return fig


def plot_training_results(training_out, figsize=(12, 6), tight_layout=True) -> plt.Figure:
    """ Plot the training results from a pandas DataFrame.

    More specifically, this function create two axes, to plot on one hand the training and validation loss over the
    epochs, and the training and validation accuracy on the other hand.

    Parameters
    ----------
    training_out : pd.DataFrame
        The output of the Trainer.fit method (see training.py). It should contain the following columns: epoch,
        train_loss, val_loss, train_accuracy, val_accuracy
    figsize : tuple
    tight_layout : bool
        If True, the figure is adjusted to the axes and labels

    Returns
    -------
    fig : plt.Figure
        The figure containing the plots
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    sns.lineplot(data=training_out, x='epoch', y='train_loss', ax=ax[0], label="Training loss")
    sns.lineplot(data=training_out, x='epoch', y='val_loss', ax=ax[0], label="Validation loss")
    sns.lineplot(data=training_out, x='epoch', y='train_accuracy', ax=ax[1], label="Training accuracy")
    sns.lineplot(data=training_out, x='epoch', y='val_accuracy', ax=ax[1], label="Validation accuracy")

    max_epo = training_out["epoch"].max()
    ax[0].set_ylabel("Loss (cross-entropy)")
    ax[0].set_xlim(1, max_epo)
    ax[0].set_xticks(range(5, max_epo + 1, 5 if max_epo > 15 else 1))
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].set_xlim(1, max_epo)
    ax[1].set_xticks(range(5, max_epo + 1, 5 if max_epo > 15 else 1))

    if tight_layout:
        fig.tight_layout()

    return fig
