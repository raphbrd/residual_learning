import os

import pandas as pd
import streamlit as st
import torch
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from stqdm import stqdm
from torch import nn
from torch.optim.lr_scheduler import StepLR

from models import xavier_weights, ResNet, PlainNet
from training import Trainer, CallBacks, Stream
from utils import get_input_size, get_device
from widgets import select_model, load_data_st, select_optimizer

st.title("Gradient Analysis of Deep Residual Learning")
st.markdown("""This is a playground where you can experiment with the gradient analysis of Convolutional Neural 
Networks (CNNs), with and without residual connections. You can either train a model on the CIFAR-10 dataset and then analyze the gradients, or load a pre-trained model to
analyze the gradients.""")
st.info("Gradient analysis is only available for models trained on the CIFAR-10 dataset.")
page_mode = st.sidebar.selectbox("Choose models to use :", [
    "Train a model with gradient tracking",
    "Pre-trained models",
])

data_path = '/Users/raphaelbordas/Code/sandbox_deep_learning/data'
train_loader, validation_loader, test_loader, classes = load_data_st(data_path, "CIFAR10", 128, 0.1)
input_size = get_input_size(train_loader)
if page_mode == "Train a model with gradient tracking":
    st.header("Train a model with gradient tracking")
    st.write("As on the Homepage, you can train a model on the CIFAR-10 dataset. This time, the gradients will be "
             " saved in the same file as the model dictionary state after each epochs. This allows to check the "
             " evolution of the gradients during training (in term of L2 norm and variance).")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        model = select_model(input_size, len(classes), models=["ResNet", "PlainNet"])
        xavier_weights(model)
    with col2:
        optimizer, scheduler, n_epochs = select_optimizer(model)
        criterion = nn.CrossEntropyLoss()

    save_path = st.text_input("Directory path to save the model (the file name will be automatically generated)",
                              value="/Users/raphaelbordas/Code/sandbox_deep_learning/projet_dl/derivatives/")
    save_desc = st.text_input("A description to add to the files to distinguish them from other runs (e.g., the number "
                              "of layers or the type of residual blocks)",
                              value="")
    if len(save_desc) > 100:
        save_desc = ""
        st.error(f"The description is too long (max. 100 characters). It will not be saved.")
    if not os.path.exists(save_path):
        st.error(f"Path {save_path} does not exist. Training is still possible but the model will not be saved.")

    training = st.button("Train the model with gradient tracking", type="primary")
    if training:
        trainer = Trainer(model, optimizer, scheduler, criterion, callbacks=CallBacks(Stream(st.write, stqdm)),
                          device=get_device(), save_path=save_path)
        training_out = trainer.fit(train_loader, validation_loader, n_epochs=n_epochs, save_epo_state=True,
                                   desc=save_desc, save_grads=True)

    load = st.button("Run gradient analysis", type="primary")
    if load:
        checkpoint = torch.load(os.path.join(save_path, "{model}_{desc}_{optim}_{dataset}_epo_{epo}.pt".format(
            model=model.__class__.__name__,
            optim=optimizer.__class__.__name__,
            dataset="CIFAR10",
            epo=n_epochs,
            desc=save_desc
        )))
        # layers of interest
        layers = [k for k in checkpoint["grads"].keys() if "conv" in k and "bias" not in k and "skip" not in k]

        grads = pd.DataFrame({
            "layer": layers,
            "norm": [np.linalg.norm(checkpoint["grads"][k].cpu().detach().flatten()) for k in layers],
            "var": [np.var(checkpoint["grads"][k].cpu().detach().flatten().numpy()) for k in layers],
        })

        plt.style.use("seaborn-v0_8-talk")
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
        layers_idx = np.arange(len(layers)) + 1
        axes[0].plot(layers_idx, grads["norm"], label=model.__class__.__name__)
        axes[1].plot(layers_idx, grads["var"], label=model.__class__.__name__)
        axes[1].set_yscale("log")
        axes[0].legend()
        axes[1].legend()
        axes[0].set_ylabel("Gradients L2 norm")
        axes[1].set_ylabel("Gradients variance (log)")
        axes[0].set_xlim(0, len(layers) + 1)
        axes[1].set_xlim(0, len(layers) + 1)
        fig.supxlabel("Layers")
        fig.tight_layout()
        st.pyplot(fig)
elif page_mode == "Pre-trained models":
    st.header("Gradient analysis of pre-trained models")
    st.write("You can load a pre-trained model and analyze the gradients of the convolutional layers.")

    # Pre-trained models should be
    # - ResNet-14
    # - PlainNet-14
    # - ResNet-38
    # - PlainNet-38

    col1, col2 = st.columns(2, gap="large")
    model_size = col1.selectbox("Select the number of layers", [14, 38])
    epochs = col2.select_slider("Select at which epoch to look at the gradients", options=[1, 5, 10], value=5)

    features = [16, 32, 64]
    if model_size == 14:
        modules = [2, 2, 2]
    elif model_size == 38:
        modules = [6, 6, 6]
    else:
        raise ValueError("Model description must be contains either 14 or 38 layers.")

    resnet = ResNet(input_size[1], len(classes), module_list=modules, features_shapes=features)
    plain = PlainNet(input_size[1], len(classes), module_list=modules, features_shapes=features)
    optimizer = torch.optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    load = st.button("Run gradient analysis", type="primary")
    if load:
        checkpoint_resnet = torch.load("./models/ResNet_{desc}_{optim}_{dataset}_epo_{epo}.pt".format(
            model=resnet.__class__.__name__,
            optim="SGD",
            dataset="CIFAR10",
            epo=epochs,
            desc=str(model_size) + "_with_grad"
        ))
        checkpoint_plain = torch.load("./models/PlainNet_{desc}_{optim}_{dataset}_epo_{epo}.pt".format(
            model=resnet.__class__.__name__,
            optim="SGD",
            dataset="CIFAR10",
            epo=epochs,
            desc=str(model_size) + "_with_grad"
        ))
        # layers of interest
        layers = [k for k in checkpoint_resnet["grads"].keys() if "conv" in k and "bias" not in k and "skip" not in k]

        grads = pd.DataFrame({
            "layer": layers,
            "plain_norm": [np.linalg.norm(checkpoint_plain["grads"][k].cpu().detach().flatten()) for k in layers],
            "resnet_norm": [np.linalg.norm(checkpoint_resnet["grads"][k].cpu().detach().flatten()) for k in layers],
            "plain_var": [np.var(checkpoint_plain["grads"][k].cpu().detach().flatten().numpy()) for k in layers],
            "resnet_var": [np.var(checkpoint_resnet["grads"][k].cpu().detach().flatten().numpy()) for k in layers]
        })

        plt.style.use("seaborn-v0_8-talk")
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
        layers_idx = np.arange(len(layers)) + 1
        axes[0].plot(layers_idx, grads["plain_norm"], label="PlainNet")
        axes[0].plot(layers_idx, grads["resnet_norm"], label="ResNet")
        axes[1].plot(layers_idx, grads["plain_var"], label="PlainNet")
        axes[1].plot(layers_idx, grads["resnet_var"], label="ResNet")
        axes[1].set_yscale("log")
        axes[0].legend()
        axes[1].legend()
        axes[0].set_ylabel("Gradients L2 norm")
        axes[1].set_ylabel("Gradients variance (log)")
        axes[0].set_xlim(0, len(layers) + 1)
        axes[1].set_xlim(0, len(layers) + 1)
        fig.supxlabel("Layers")
        fig.tight_layout()
        st.pyplot(fig)