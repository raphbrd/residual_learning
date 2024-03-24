import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR

from models import *
from data import _load_torch_data
from training import CallBacks, Stream, Trainer, analyze_gradients
from utils import get_input_size, count_parameters
from viz import plot_training_results

# path is used to save the results
path = "/Users/raphaelbordas/Code/sandbox_deep_learning/projet_dl/derivatives/grad_analysis"
n_epochs = 10
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
desc = "38"  # for the name of the saved files
modules = [6, 6, 6]
features = [16, 32, 64]
plain = PlainNet(input_size[1], len(classes), module_list=modules, features_shapes=features)
resnet = ResNet(input_size[1], len(classes), module_list=modules, features_shapes=features)
criterion = nn.CrossEntropyLoss()

if init_weights == "xavier":
    xavier_weights(plain)
    xavier_weights(resnet)
elif init_weights == "kaiming":
    kaiming_weights(plain)
    kaiming_weights(resnet)

for model in [plain, resnet]:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

    trainer = Trainer(model, optimizer, scheduler, criterion, callbacks=CallBacks(Stream()),
                      device=torch.device("mps"), save_path=path)
    training_out = trainer.fit(train_loader, validation_loader, n_epochs=n_epochs, save_epo_state=True,
                               desc=desc + "_with_grad", save_grads=True)

    fig = plot_training_results(training_out)
    fig.savefig(path + f"/{model.__class__.__name__}{desc}_training_results.png")

    test_loss, accuracy = trainer.run_test(test_loader)
    out = trainer.run_test_per_class(test_loader)
    last_row = pd.DataFrame(
        {"classes": "all", "accuracy": accuracy, "loss": test_loss, "n_instances": len(test_loader.dataset)}, index=[0])
    out = pd.concat([out, last_row], ignore_index=True)
    out.to_csv(path + f"/{model.__class__.__name__}{desc}_results_test_per_class.csv")

    print(f"{model.__class__.__name__} network done.")
