# Residual Networks for (very) deep learning in computer vision

Project made during the first year of the
[Mathematics and AI Master at Paris-Saclay University](https://www.universite-paris-saclay.fr/en/education/master/mathematics-and-applications/m1-mathematiques-et-intelligence-artificielle).
This repository contains a Streamlit app and a playground for Residual Networks, notably a PyTorch implementation of He
al. 2016a.

Author: Raphael Bordas

## Installation

## Usage

- Run the Streamlit app with `streamlit run Homepage.py`
- Run the sandbox with `python sandbox.py`

The Streamlit app allows you to train a ResNet on CIFAR-10 and visualize the training process but is quite limited
regarding the gradient analysis. Pre-train models allows comparison between ResNets and PlainNets (their plain network
counterparts without residual connections).
To perform more advanced analysis, you can use the sandbox with the `save_grads` of the
Trainer enabled. The notebook `Gradient Analysis.ipynb` provides a typical analysis of such outputs.

## Code organization

Main files :

- `Homepage.py`: Streamlit app
    - 2 other pages are available in the app :
        - `pages/1_Gradient_Analysis.py` : Gradient analysis oriented page, either on pre-trained models or on online
          trained models with gradient tracking
        - `pages/2_Motivation.py` : A simple example on how increasing depth can affect accuracy on simple dataset such
          as MNIST
- `sandbox.py`: Playground for ResNet. Mainly used for debugging, testing and pre-training models.
- `models.py`: PyTorch implementation of ResNet, PlainNets, Multi-layer Perceptron and a simple CNN (LeNet-5).
- `training.py`: Training utilities. The main class is `Trainer` which is used to train models, load, save state
  dictionaries, etc.

Notebooks :

- `Gradient Analysis.ipynb`: Example of gradient analysis on a pre-trained model.
- `Figures.ipynb`: Figures from the report
- `Experimentation.ipynb`: Code to run the training from Kaggle or Google Colab and access a cuda GPU.

Utilities :

- `layers.py` : Implementation of the basic layers used in the models.
- `residuals.py` : Contains the implementation of the ResNet blocks (ConvBlock).
- `data.py`: Wrapper functions and classes for convenient PyTorch datasets loading.
- `utils.py`, `viz.py`: Various utilities for visualization, reports, etc.