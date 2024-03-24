import os.path

import streamlit as st
import torch
from stqdm import stqdm
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary

from data import file_list_in_directory
from models import LeNet5, DeepMLP, ResNet, PlainNet, xavier_weights, kaiming_weights
from residuals import ConvResBlock, ConvResBlockPre
from training import Trainer, CallBacks, Stream
from utils import get_input_size
from viz import plot_training_results
from widgets import load_data_st, display_datasets_metrics, set_dataset_sidebar, display_sample_img, select_model, \
    select_optimizer

st.set_page_config(page_title="Deep Learning Sandbox", layout="wide")
st.title('Residual Networks for deep learning')

# TODO : remove MNIST from the list of datasets, it is only useful on the motivation page
# TODO : add a streamlit input for the data path
dataset_name, batch_size, validation_size = set_dataset_sidebar(st)
data_path = '/Users/raphaelbordas/Code/sandbox_deep_learning/data'
train_loader, validation_loader, test_loader, classes = load_data_st(data_path, dataset_name, batch_size,
                                                                     validation_size)

st.markdown("""This is a project aiming at providing a simple and interactive environment to experiment with deep 
learning models, more particularly with residual networks. The main page is a sandbox where you can train a model on 
the MNIST or CIFAR-10 dataset, and then visualize the results. The second page is a playground where you can 
experiment with the gradient analysis of simple Multi-Layer Perceptron (MLP) and Convolutional Neural Networks 
(CNNs), with and without residual connections.
""")
st.header("Sandbox for deep learning")

st.subheader(f"Characteristics of the {dataset_name} dataset")
# Notify the user that the data was successfully loaded.
data_load_state = st.info(f'Loading {dataset_name} data (this can take some time...)')
data_load_state.success(f"{dataset_name} Data loaded!")

display_datasets_metrics(st, train_loader, validation_loader, test_loader, classes)
display_sample_img(train_loader, classes, dataset_name)

st.subheader("Training a model on the dataset")

# 1. Choosing and tuning the model and the optimizer
# TODO : refactor this with the functions in the widgets.py file
input_size = get_input_size(train_loader)
col1, col2 = st.columns(2, gap="large")
with col1:
    model = select_model(input_size, len(classes))

with col2:
    st.markdown("**Optimizer and training parameters**")
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler, n_epochs = select_optimizer(model)

    # Initialize the weights of the model
    init_weights = st.radio("Initialize the weights of the model with:",
                            options=["gaussian Xavier", "gaussian Kaiming", "default"],
                            help="Here bias undergoes the same init. as the weights. "
                                 "PyTorch default is uniform Kaiming.")
    if init_weights == "gaussian Xavier":
        xavier_weights(model)
    elif init_weights == "gaussian Kaiming":
        kaiming_weights(model)

with st.expander("Model architecture", expanded=True):
    st.text(summary(
        model, input_size=input_size, col_names=("input_size", "output_size", "num_params"), verbose=0, depth=5
    ))

# 2. Set up the training parameters
save = st.radio("Save the state of the model and the optimizer:",
                options=["Never", "At the end of training", "At each epoch"], )
save_path = None
save_desc = ""
if save == "Never":
    st.warning("The model will not be saved and re-run of the"
               " streamlit app will discard any trained model and its results.")
else:
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
        save = "Never"
        save_path = None

# 3. Train the model
trainer = Trainer(model, optimizer, scheduler, criterion, callbacks=CallBacks(Stream(st.write, stqdm)),
                  device=torch.device("mps"), save_path=save_path)
is_training = st.button(f"Train the **{model.__class__.__name__}** model", type="primary")
if is_training:
    with st.status("Training the model...", expanded=True) as status:
        training_out = trainer.fit(train_loader, validation_loader, n_epochs=n_epochs,
                                   save_epo_state=save == "At each epoch", desc=save_desc)

    st.write(training_out)
    if n_epochs > 1:
        st.pyplot(plot_training_results(training_out))
    else:
        st.warning("No other training results to display for a single epoch.")
    test_loss, accuracy = trainer.run_test(test_loader)
    st.success(f"**Performance on the test set:** Average loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
    out = trainer.run_test_per_class(test_loader)
    # TODO: should the table and the results be cached for display even if there are changes elsewhere on the page?
    st.write(out)

# 4. Look at saved models
st.subheader("Loading models")

st.markdown("""How to load a model:
1. Create the model and the optimizer with the same hyperparameters as the model you want to load.
2. Choose the model file to load.
4. Set up the model : the created model will have its weights and optimizer state updated with 
the ones from the loaded model.
""")
st.info("If there are any differences between the model from the file and the model created above "
        "in the \"Model creation\" section, an error will be raised.")

load_path = st.text_input("Directory path to load the models from:",
                          value="/Users/raphaelbordas/Code/sandbox_deep_learning/projet_dl/derivatives/")
if not os.path.exists(load_path):
    st.error(f"Path {load_path} does not exist. Cannot load any model.")
    load_path = None

if load_path is not None:
    all_files = file_list_in_directory(load_path)

    # the select uses the first column of the dataframe as the options
    current_model_fname = st.selectbox("File to load", all_files)
    st.write(f"Current model file: **{current_model_fname}**,"
             f" created on **{all_files[all_files['Name'] == current_model_fname].iloc[0, 1]}**.")
    setup_model = st.button("Setup the model", type="primary")
    model_checkpoint = dict()
    if setup_model:
        train_checkpoint = trainer.load_state(load_path + current_model_fname)
        st.success("Model setup completed!")
        out = trainer.run_test_per_class(test_loader)
        st.write(out)

# TODO add and test CNN support for ResNets
# TODO display the results of the test per class, not only the overall accuracy

# TODO set up a new page for gradient analysis of Perceptron (with and without residual connections)
# TODO allow multiple kind of analysis (e.g. gradient norm, gradient variance, etc.)
# TODO allow multiple kind of residual connections and ResNets
# TODO pre-implement classical models (ResNet-18, LeNet-5, etc.) and allow the user to tweak them
