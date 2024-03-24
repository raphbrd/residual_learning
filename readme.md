# Residual Networks for (very) deep learning in computer vision

Project made during the first year of the
[Mathematics and AI Master at Paris-Saclay University](https://www.universite-paris-saclay.fr/en/education/master/mathematics-and-applications/m1-mathematiques-et-intelligence-artificielle).
This repository contains a Streamlit app and a playground for Residual Networks, notably a PyTorch implementation of He
al. 2016a.

Author: Raphael Bordas

## Installation

## Usage

- Run the Streamlit app with `streamlit run app.py`
- Run the sandbox with `python sandbox.py`

The Streamlit app allows you to train a ResNet on CIFAR-10 and visualize the training process but is quite limited
regarding the gradient analysis.

To perform more advanced analysis, you can use the sandbox with the `save_grads` of the Trainer enabled. The notebook
`Gradient Analysis.ipynb` provides a typical analysis of such outputs.
