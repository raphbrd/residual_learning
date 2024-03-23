import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR

from models import *
from data import _load_torch_data
from residuals import ConvResBlockPre, ConvPlainBlock
from training import CallBacks, Stream, Trainer
from utils import get_input_size, count_parameters
from viz import plot_training_results

# path is used to save the results
path = "/Users/raphaelbordas/Code/sandbox_deep_learning/projet_dl/derivatives"
n_epochs = 20
batch_size = 128
val_size = 0.1
init_weights = "xavier"

# training
train_loader, validation_loader, test_loader, classes = _load_torch_data(
    data_path='/Users/raphaelbordas/Code/sandbox_deep_learning/data',
    dataset_name="CIFAR10",
    batch_size=batch_size,
    val_size=val_size)

input_size = get_input_size(train_loader)
lenet = LeNet5(input_size[1], len(classes), input_size[-1])
# plain20 = ResNet(input_size[1], len(classes),
#                  module_list=[3, 3, 3],
#                  features_shapes=[16, 32, 64],
#                  block_type=ConvPlainBlock)
resnet20 = ResNet(input_size[1], len(classes),
                  module_list=[3, 3, 3],
                  features_shapes=[16, 32, 64],
                  block_type=ConvResBlock)
# resnet20_pre = ResNet(input_size[1], len(classes),
#                       module_list=[3, 3, 3],
#                       features_shapes=[16, 32, 64],
#                       block_type=ConvResBlockPre)
# resnet38 = ResNet(input_size[1], len(classes),
#                   module_list=[6, 6, 6],
#                   features_shapes=[16, 32, 64],
#                   block_type=ConvResBlock)
print(f"LeNet5 :\t{count_parameters(lenet, verbose=False)[1]}")
# print(f"Plain-20 :\t{count_parameters(plain20, verbose=False)[1]}")
print(f"ResNet-20 :\t{count_parameters(resnet20, verbose=False)[1]}")
# print(f"ResNet-20 pre-activated :\t{count_parameters(resnet20_pre, verbose=False)[1]}")
# print(f"ResNet-38 :\t{count_parameters(resnet38, verbose=False)[1]}")

# selecting the model to train here :
model = resnet20

if init_weights == "xavier":
    xavier_weights(model)
elif init_weights == "kaiming":
    kaiming_weights(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
criterion = nn.CrossEntropyLoss()

trainer = Trainer(model, optimizer, scheduler, criterion, callbacks=CallBacks(Stream()),
                  device=torch.device("mps"), save_path=path)
training_out = trainer.fit(train_loader, validation_loader, n_epochs=n_epochs, save_epo_state=True)

fig = plot_training_results(training_out)
if path is not None:
    fig.savefig(path + "/training_results.png")
plt.show()

test_loss, accuracy = trainer.run_test(test_loader)
print(test_loss, accuracy)
out = trainer.run_test_per_class(test_loader)
print(out)
last_row = pd.DataFrame(
    {"classes": "all", "accuracy": accuracy, "loss": test_loss, "n_instances": len(test_loader.dataset)}, index=[0])
out = pd.concat([out, last_row], ignore_index=True)

if path is not None:
    out.to_csv(path + "/results_test_per_class.csv")
