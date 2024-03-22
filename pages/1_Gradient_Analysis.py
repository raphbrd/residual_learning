import streamlit as st

st.title("Gradient Analysis of Deep Residual Learning")

# TODO : add a dropdown menu to select the model to analyze
# TODO : create a class for a separate model saver/loader, common to this page and the main page
st.markdown("""This is a playground where you can experiment with the gradient analysis of simple Multi-Layer 
Perceptron (MLP) and Convolutional Neural Networks (CNNs), with and without residual connections. It is recommended to
first train a model on the MNIST or CIFAR-10 dataset in the main page, and then come here to analyze the gradients. Or to
load a pre-trained model.

To do so, you can select a model from the dropdown menu, and then choose the layer to analyze. The gradients are then
displayed in a heatmap, where the x-axis represents the input features and the y-axis the output features of the
selected layer. The color intensity represents the magnitude of the gradients. 
""")

# TODO implement the gradient analysis