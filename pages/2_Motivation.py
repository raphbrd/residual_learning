import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import torch
from matplotlib import pyplot as plt
from stqdm import stqdm
from torch import optim, nn

from models import DeepMLP
from training import Trainer, CallBacks, Stream
from utils import get_input_size, get_device
from widgets import display_col_sliders, load_data_st

st.title("Motivation for deep residual learning")

st.header("Vanishing gradients problem")
st.markdown("""Illustration of the vanishing gradients problem in deep neural networks through the example
of a simple multi-layer perceptron (MLP) trained on the MNIST dataset.

This playground will generate a simple MLP with an increasing number of hidden layers, one model per depth. As an
example, MLPs from 1 to 8 hidden layers with constant size of 128 neurons per layers are already trained. All models
considered on the pages also have a first layer of the input dimension and a last one of the output dimension (10 
classes with a softmax activation function).
""")
device = get_device()
data_path = '/Users/raphaelbordas/Code/sandbox_deep_learning/data'
train_loader, validation_loader, test_loader, classes = load_data_st(data_path, "MNIST", 512)
input_size = get_input_size(train_loader)

page_mode = st.sidebar.selectbox("Choose models to use :", [
    "Pre-trained models",
    "Train a new model",
], index=0)

if page_mode == "Pre-trained models":
    st.header("Pre-trained models")
    st.write("The following models are already trained and can be loaded to check the accuracy on the test set in "
             "function of the number of hidden layers. All models have a first layer of the input dimension and a "
             "last one of the output dimension (10 classes with a softmax activation function). In between, there "
             "are n layers of 128 units. ")
    data_load_state = st.info(f'Computing the accuracy of the models on the test set (this might take a minute)...')
    df = pd.DataFrame(np.zeros((10, 2)), columns=["Hidden layers", "Accuracy"])
    for i in range(1, 11):
        layers = [64] + [128] * i + [64]
        model = DeepMLP(layers=[input_size[2] ** 2 * input_size[1]] + layers + [len(classes)])
        optimizer = optim.Adam(model.parameters())
        trainer = Trainer(model, optimizer, scheduler=None, criterion=nn.CrossEntropyLoss(),
                          callbacks=None,
                          device=get_device(), save_path="./models")
        model, _, _ = trainer.load_state(f"./models/DeepMLP_MLP_{i + 2}_layers_Adam_MNIST_epo_20.pt")
        test_loss, accuracy = trainer.run_test(test_loader)
        df.iloc[i - 1] = [i, accuracy]
    data_load_state.success(f"Computation done!")
    df["Hidden layers"] = df["Hidden layers"] + 2
    st.write(df)
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(df, x="Hidden layers", y="Accuracy", ax=fig.gca())
    st.pyplot(fig)
    st.success("We clearly see the degradation of the accuracy!")

elif page_mode == "Train a new model":
    st.info("The models will be trained over 30 epochs to ensure proper convergence.")
    n_hidden_layers, size_hidden_layers = display_col_sliders(st, ("Number of hidden layers", 1, 15, 1),
                                                              ("Size of the hidden layers", 16, 256, 64, 16))
    layers = [size_hidden_layers]
    df = pd.DataFrame(np.zeros((n_hidden_layers, 2)), columns=["Hidden layers", "Accuracy"])
    is_training = st.button(f"Train MLP", type="primary")
    if is_training:
        for mdl in range(1, n_hidden_layers + 1):
            layers.append(size_hidden_layers)
            model = DeepMLP(layers=[input_size[2] ** 2 * input_size[1]] + layers + [len(classes)])
            # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            optimizer = optim.Adam(model.parameters())
            trainer = Trainer(model, optimizer, scheduler=None, criterion=nn.CrossEntropyLoss(),
                              callbacks=CallBacks(Stream(st.write, stqdm)),
                              device=get_device(), save_path=None)
            with st.status(f"Training model {mdl}...", expanded=True) as status:
                training_out = trainer.fit(train_loader, None, n_epochs=30, save_epo_state=False)
                status.update(label=f"Training of model {mdl} completed!", state="complete", expanded=False)
            # st.write(training_out)
            test_loss, accuracy = trainer.run_test(test_loader)
            df.iloc[mdl - 1] = [mdl, accuracy]
        st.write(df)
        fig = plt.figure(figsize=(12, 6))
        sns.barplot(df, x="Hidden layers", y="Accuracy", ax=fig.gca())
        st.pyplot(fig)
