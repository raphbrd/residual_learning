import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import torch
from matplotlib import pyplot as plt
from stqdm import stqdm
from torch import optim, nn

from data import load_data_st
from models import DeepMLP
from training import Trainer, CallBacks, Stream
from utils import get_input_size
from widgets import display_col_sliders

st.title("Motivation for deep residual learning")

st.header("Vanishing gradients problem")
st.markdown("""Illustration of the vanishing gradients problem in deep neural networks through the example
of a simple multi-layer perceptron (MLP) trained on the MNIST dataset. Use the different options below to 
observe the effects of existing strategies to mitigate the vanishing gradients problem.
""")
# TODO: build a simple MLP with increasing number of layers
# TODO : plot the accuracy on the MNIST dataset after one epoch (lr = 0.1 for speed purposes)
# TODO : plot the gradients of the weights of the first layer
# TODO : add possibility of using ReLU or Sigmoid as activation function
# TODO : add possibility of using batch norm and different initialization schemes

device = torch.device("mps")

train_loader, validation_loader, test_loader, classes = load_data_st("MNIST", 512)
input_size = get_input_size(train_loader)

# n_hidden_layers = st.slider("Number of hidden layers", 1, 10, 1)
# size_hidden_layers = st.slider("Size of the hidden layers", 16, 256, 32, step=16)
n_hidden_layers, size_hidden_layers = display_col_sliders(st, ("Number of hidden layers", 1, 10, 1),
                                                          ("Size of the hidden layers", 16, 256, 32, 16))
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
                          device=torch.device("mps"), save_path=None)
        with st.status(f"Training the model {mdl}...", expanded=True) as status:
            training_out = trainer.fit(train_loader, None, n_epochs=30, save_epo_state=False)
            status.update(label=f"Training of model {mdl} completed!", state="complete", expanded=False)
        # st.write(training_out)
        test_loss, accuracy = trainer.run_test(test_loader)
        df.iloc[mdl - 1] = [mdl, accuracy]
    st.write(df)
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(df, x="Hidden layers", y="Accuracy", ax=fig.gca())
    st.pyplot(fig)
