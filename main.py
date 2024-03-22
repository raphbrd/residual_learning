import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR
from torch import nn, optim, functional as F

from models import *
from data import _load_torch_data
from training import CallBacks, Stream, Trainer
from utils import get_input_size
from viz import plot_training_results

# handling arguments
parser = argparse.ArgumentParser(description="Train a ResNet model on CIFAR10")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
parser.add_argument("--val_size", type=float, default=0.10, help="Validation set size")
parser.add_argument("--n_epochs", type=int, default=30, help="Number of epochs to train the model")
parser.add_argument("--init_weights", type=str, default="xavier", choices=["xavier", "kaiming", "default"],
                    help="Initialization method for the weights")
parser.add_argument("--path", type=str, default=None, help="Path to save the model and the derivatives")
args = parser.parse_args()

path = args.path
if path is not None and not os.path.exists(path):
    raise ValueError(f"Path {path} does not exist.")

n_epochs = args.n_epochs
batch_size = args.batch_size
val_size = args.val_size
init_weights = args.init_weights

# training
train_loader, validation_loader, test_loader, classes = _load_torch_data(
    data_path='/Users/raphaelbordas/Code/sandbox_deep_learning/data',
    dataset_name="CIFAR10",
    batch_size=batch_size,
    val_size=val_size)

input_size = get_input_size(train_loader)
resnet20 = ResNet(input_size[1], len(classes),
                  module_list=[3, 3, 3],
                  features_shapes=[16, 32, 64],
                  block_type="ConvResBlock")
if init_weights == "xavier":
    xavier_weights(resnet20)
elif init_weights == "kaiming":
    kaiming_weights(resnet20)

optimizer = torch.optim.SGD(resnet20.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.CrossEntropyLoss()

trainer = Trainer(resnet20, optimizer, scheduler, criterion, callbacks=CallBacks(Stream()),
                  device=torch.device("mps"), save_path=path)
training_out = trainer.fit(train_loader, validation_loader, n_epochs=n_epochs, save_epo_state=False)

fig = plot_training_results(training_out)
if path is not None:
    fig.savefig(path + "/training_results_plain38.png")
plt.show()

test_loss, accuracy = trainer.run_test(test_loader)
print(test_loss, accuracy)
out = trainer.run_test_per_class(test_loader)
if path is not None:
    out.to_csv(path + "/restuls_test_per_class_plain38.csv")
print(out)
