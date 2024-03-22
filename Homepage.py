import os.path

import streamlit as st
import torch
from stqdm import stqdm
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary

from data import file_list_in_directory
from models import LeNet5, DeepMLP, ResMLP, ResNet
from training import Trainer, CallBacks, Stream
from utils import get_input_size
from viz import plot_training_results
from widgets import load_data_st, display_datasets_metrics, set_dataset_sidebar, display_sample_img

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
    st.markdown("**Model creation**")
    model_name = st.selectbox("Model", ["LeNet5", "MLP", "ResMLP", "ResNet"], index=0)
    # default model is LeNet5 : input size is of shape (batch_size, n_channels, width, height)
    model = LeNet5(input_size[1], len(classes), image_width=input_size[2])
    if model_name == "MLP":
        choices = [64, 128, 256, 512, 1024, 2048] * 3
        layers = st.multiselect("Number of units for each hidden layer", choices, default=[128, 64])
        model = DeepMLP(layers=[input_size[2] ** 2 * input_size[1]] + layers + [len(classes)])
    elif model_name == "ResMLP":
        n_res_blocks = st.slider("Number of residual blocks", 1, 10, 4)
        tabs = st.tabs([f"ResBlock {i + 1}" for i in range(n_res_blocks)])
        shapes = [(0, 0, 0)] * n_res_blocks
        for idx, tab in enumerate(tabs):
            with tab:
                cols = st.columns(3)
                input_shape = cols[0].slider(f"Input shape", 64, 1024, 128, step=64, key=f"input_shape_{idx}")
                hidden_shape = cols[1].slider(f"Hidden shape", 64, 1024, 256, step=64, key=f"hidden_shape_{idx}")
                output_shape = cols[2].slider(f"Output shape", 64, 1024, 128, step=64, key=f"output_shape_{idx}")
                shapes[idx] = (input_shape, hidden_shape, output_shape)
        model = ResMLP(input_size[2] ** 2 * input_size[1], len(classes), n_blocks=n_res_blocks, block_shapes=shapes)
        try:
            x = model(torch.randn(*input_size))
        except RuntimeError as e:
            st.error(f"Error in the forward pass of the model with those parameters: {e}.")
            model = None
    elif model_name == "ResNet":
        # TODO: explain to the use what is a module (= a set of residual blocks with the same parameters, as it is
        # common for ResNets to increase layers depth by duplicating residual blocks). A residual blocks of a
        # standard resnet is composed of 2 convolutional layers with batch normalization and ReLU activation.
        st.info("A residual module is a set of residual blocks with identical parameters. "
                "Each block performs two 3x3 convolutions with the specified output channels. "
                "When required, a projection is automatically added for the identity mapping. "
                "See the report for details.")
        n_res_modules = st.slider("Number of residual modules", 1, 3, 2)
        shapes = [tuple()] * n_res_modules
        features_shapes = [16] * n_res_modules
        tabs = st.tabs([f"Module {i + 1}" for i in range(n_res_modules)])
        for idx, tab in enumerate(tabs):
            with tab:
                block_type = st.selectbox("Block type", ["ConvResBlock", "BottleNeck"], index=0,
                                          key=f"block_type_{idx}")
                subcol1, subcol2 = st.columns(2, gap="medium")
                n_blocks = subcol1.slider(f"Number of blocks", 1, 6, 2, key=f"n_blocks_{idx}")
                out_channels = subcol2.slider(f"Output channels for each 3x3conv", 6, 128, 16, step=2,
                                              key=f"n_chs_in_{idx}")
                shapes[idx] = n_blocks
                features_shapes[idx] = out_channels
        model = ResNet(input_size[1], len(classes), module_list=shapes, features_shapes=features_shapes)

with col2:
    st.markdown("**Optimizer and training parameters**")
    optimizer_name = st.selectbox("Optimizer", ["SGD", "Adam"],
                                  format_func=lambda lab: dict(SGD="SGD with momentum", Adam="Adam")[lab], index=0,
                                  disabled=True, help="Most papers uses SGD.")
    learning_rate = st.select_slider("Learning rate (LR)", options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                                     value=0.01)
    columns = st.columns(2, gap="medium")
    # st.info("If either of these LR decay parameters is set to 0, no decay will be performed.")
    scheduler_lr_step = columns[0].slider("LR decay every $n$ epoch", 0, 50, value=10, step=5, help="0 means no decay")
    scheduler_lr_gamma = columns[1].select_slider("LR decay scale", [0, 0.0001, 0.001, 0.01, 0.1], value=0.1,
                                                  help="0 means no decay")
    n_epochs = st.slider("Number of epochs", 1, 50, value=15)

if model is not None:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = None
    if scheduler_lr_gamma > 0 and scheduler_lr_step > 0:
        scheduler = StepLR(optimizer, step_size=scheduler_lr_step, gamma=scheduler_lr_gamma)
criterion = nn.CrossEntropyLoss()

# TODO: the summary function does not display the skip connection and the relu at the end of each residual block
if model is not None:
    with st.expander("Model architecture", expanded=True):
        st.text(summary(
            model, input_size=input_size, col_names=("input_size", "output_size", "num_params"), verbose=0, depth=5
        ))

# 2. Set up the training parameters
save = st.radio(
    "Save the state of the model and the optimizer:",
    options=["Never", "At the end of training", "At each epoch"],
)
save_path = None
if save == "Never":
    st.warning("The model will not be saved and re-run of the"
               " streamlit app will discard any trained model and its results.")
else:
    save_path = st.text_input("Directory path to save the model (the file name will be automatically generated)",
                              value="/Users/raphaelbordas/Code/sandbox_deep_learning/projet_dl/derivatives/")
    if not os.path.exists(save_path):
        st.error(f"Path {save_path} does not exist. Training is still possible but the model will not be saved.")
        save = "Never"
        save_path = None

# 3. Train the model
if model is not None:
    is_training = st.button(f"Train the **{model.__class__.__name__}** model", type="primary")
    if is_training:
        with st.status("Training the model...", expanded=True) as status:
            trainer = Trainer(model, optimizer, scheduler, criterion, callbacks=CallBacks(Stream(st.write, stqdm)),
                              device=torch.device("mps"), save_path=save_path)
            training_out = trainer.fit(train_loader, validation_loader, n_epochs=n_epochs,
                                       save_epo_state=save == "At each epoch")
            status.update(label="Training completed!", state="complete", expanded=False)

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
st.subheader("Saving and loading models")

load_path = st.text_input("Directory path to load the models from:",
                          value="/Users/raphaelbordas/Code/sandbox_deep_learning/projet_dl/derivatives/")
if not os.path.exists(load_path):
    st.error(f"Path {load_path} does not exist. Cannot load any model.")
    load_path = None

if load_path is not None:
    all_files = file_list_in_directory(load_path)

    # the select uses the first column of the dataframe as the options
    current_model_fname = st.selectbox("Choose a model to load", all_files)
    st.write(
        f"Current model file: **{current_model_fname}**,"
        f" created on **{all_files[all_files['Name'] == current_model_fname].iloc[0, 1]}**.")
    st.write("The model will be set up from the model and optimizer defined above.")
    setup_model = st.button("Setup the model", type="primary")
    model_checkpoint = dict()
    if setup_model:
        train_checkpoint = Trainer.load_state(load_path, current_model_fname, model, optimizer, return_trainer=True)
        st.success("Model setup completed!")
        out = train_checkpoint.run_test_per_class(test_loader)
        st.write(out)

# TODO add and test CNN support for ResNets
# TODO display the results of the test per class, not only the overall accuracy

# TODO set up a new page for gradient analysis of Perceptron (with and without residual connections)
# TODO allow multiple kind of analysis (e.g. gradient norm, gradient variance, etc.)
# TODO allow multiple kind of residual connections and ResNets
# TODO pre-implement classical models (ResNet-18, LeNet-5, etc.) and allow the user to tweak them
