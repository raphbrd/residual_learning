from __future__ import annotations
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from torch.utils import data as torch_data
from torch import nn, optim
from tqdm import tqdm

from data import get_classes_labels

# to simplify the handling of the callbacks (= function to log messages during the training,
# either in the console or in a streamlit app), the allowed methods are hard-coded here
ALLOWED_CALLBACKS = [
    "on_train_start",
    "on_train_end",
    "on_train_batch_start",
    "on_train_batch_end",
    "on_train_epoch_start",
    "on_train_epoch_end",
    "on_save_state"
]


def get_dataset_name(loader):
    """ Retrieve the dataset from a torch.utils.data.DataLoader object. This utility function is useful to get the
    dataset from a Subset object, which is a common pattern when creating a validation set from a training set.
    """

    # if the dataset is a subset, we want to get the name of the original dataset
    if isinstance(loader.dataset, torch_data.dataset.Subset):
        return loader.dataset.dataset.__class__.__name__

    return loader.dataset.__class__.__name__


class Stream:
    """ A stream object should be able to write messages and update a progress bar on the desired output stream
    (e.g., console, file, streamlit application, etc.).

    Attributes
    ----------
    writer : callable
        A callable object that writes a message to the stream (e.g., print function, st.write)
    progress_bar_cls : class
        A class that creates a progress bar (e.g., tqdm, stqdm). The class should be passed to the constructor, not
        the instance of the class.
    """

    def __init__(self, writer=None, progress_bar_class=None):
        # writer should be a callable
        if writer is None:
            writer = print

        if progress_bar_class is None:
            progress_bar_class = tqdm

        self.writer = writer

        # the progress bar is set up through the set_pg_bar method
        self.progress_bar_cls = progress_bar_class
        self.progress_bar = None  # self.progress_bar_cls(total=1, leave=self.progress_bar_cls.__name__ == "tqdm")

    def write(self, message):
        """ write a message to the stream (e.g., console, file, streamlit application, etc.)"""
        self.writer(message)

    def update_pg_bar(self, n=1):
        """ update the progress bar by n steps """
        if self.progress_bar is not None:
            self.progress_bar.update(n)

    def set_pg_bar(self, total):
        """ set up the progress bar by instantiating the progress_bar_cls class with the total number of steps

        Parameters
        ----------
        total : int
            Total number of steps of the progress bar (= number of planned updates through the update_pg_bar method)
        """
        self.progress_bar = self.progress_bar_cls(total=total, leave=self.progress_bar_cls.__name__ == "tqdm")

    def close_pg_bar(self):
        self.progress_bar.close()


class CallBacks:
    """ Class to handle the callbacks during the training of a PyTorch model

    The methods are called at different stages of the training, such as at the start and end of the training. The
    allowed methods are defined in the ALLOWED_CALLBACKS global variable. Before adding any new method, make sure to
    add it to the ALLOWED_CALLBACKS list.
    """

    def __init__(self, stream: Stream = None):
        """ Initialize the callbacks with a stream object to write messages and update a progress bar. If no stream
        object is provided, assumes the IO operations will go through the console. """
        if stream is None:
            stream = Stream()  # default Stream IO with print and tqdm
        self.stream = stream

    def on_train_start(self):
        pass

    def on_train_end(self, df_out):
        """ Write a message at the end of the training """
        training_time = round(df_out['epoch_time'].sum())  # in seconds
        training_time_min = training_time // 60
        training_time_sec = training_time % 60
        self.stream.write(f"Training time : {training_time_min} min {training_time_sec} sec.")
        self.stream.write(f"Last epoch validation accuracy : {df_out['val_accuracy'].iloc[-1]}")
        self.stream.write(f"Best validation accuracy : {df_out['val_accuracy'].max()}, "
                          f"on epoch {df_out['val_accuracy'].idxmax() + 1}")

    def on_train_batch_start(self):
        pass

    def on_train_batch_end(self):
        """ Update the progress bar after each processed batch """
        self.stream.update_pg_bar(1)

    def on_train_epoch_start(self, total=None):
        """ Set the progress bar at the start of each epoch """
        self.stream.set_pg_bar(total)

    def on_train_epoch_end(self, epoch, n_epochs=0, epoch_loss=None, running_val_loss=None, misc=None):
        """ Write the epoch loss and the validation loss at the end of each epoch """
        self.stream.close_pg_bar()
        msg = f'Epoch {epoch + 1} / {n_epochs} | Loss: {epoch_loss} | Validation loss: {running_val_loss}'
        if misc is not None:
            msg += f" | {misc}"
        self.stream.write(msg)

    def on_save_state(self, msg):
        """ Write a message when saving the model state. This useful to remind the paths of the saved models and$
         track when the model was saved. """
        self.stream.write(msg)


class Trainer:
    """ Training class with callbacks (hooks that provide easy console logging) for a PyTorch model
    when performing a classification task.

    The training is performed by looping multiple times over the training data (n_epochs). The validation step is either
    performed at the end of each epoch or not at all (if no validation data is provided). This class allows to handle
    model saving and loading.
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler = None,
                 criterion=None, callbacks=None, device="cpu", save_path=None):
        """ Initialize the Trainer with the model and the optimizer. Data will be passed by in the fit method.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train
        optimizer : torch.optim.Optimizer
            Usually SGD or Adam optimizer
        scheduler : torch.optim.lr_scheduler.LRScheduler
            Learning rate scheduler. If None (default), no learning rate scheduler is used.
        criterion : torch.nn.Functional
            Loss function
        callbacks : CallBacks
            Functions to call at different stages of the training
        device: torch.device | str
            Device to use. Default is "cpu"
        save_path: str
            Path to save the model state and other outputs during the training. Default is None, which means no saving.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.callbacks = callbacks
        self.device = device

        self.save_path = pathlib.Path(save_path) if save_path is not None else None
        self.model_fname_template = "{model}{desc}_{optim}_{dataset}_epo_{epo}.pth"
        self.csv_fname_template = "{model}{desc}_{optim}_{dataset}_output.csv"

    def fit(self, train_loader: torch_data.DataLoader, validation_loader: torch_data.DataLoader = None, n_epochs=10,
            save_epo_state=False, desc=None) -> pd.DataFrame:
        """ Run the training loop for the specified number of epochs and return training statistics per epoch.

        The output dataframe is saved is the save_path is not None (set in the Trainer constructor).

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Training data loader
        validation_loader : torch.utils.data.DataLoader | None
            Validation data loader. Default is None, which means no validation step is performed at the end of each
            epoch.
        n_epochs : int
            Total number of epochs to train the model
        save_epo_state : bool
            If True, the model state is saved at the end of each epoch. If False and Trainer.save_path is None, no
            model state is saved. If False and Trainer.save_path is not None, the model state is saved at the end
            of the last epoch. Default is False.
        desc: str
            An optional suffix description to add to the file name if model saving was enabled.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the training and validation statistics (loss, accuracy) and the training time per epoch
        """
        if save_epo_state and self.save_path is None:
            raise ValueError("save_epo_state was to True but save_path is None."
                             "Specify save_path in the constructor or set save_epo_state to False.")

        self._handle_callback("on_train_start")

        df_epochs = []
        for epoch in range(n_epochs):
            epo_start_time = time.time()
            self._handle_callback("on_train_epoch_start", total=len(train_loader))

            # train and validate the model
            running_epoch = self.train_epoch(train_loader, epoch_idx=epoch)
            val_epoch = self.validation_step(validation_loader)
            running_epoch = pd.concat([running_epoch, val_epoch], axis=1, ignore_index=True)

            # if relevant update the learning rate scheduler at the end of each epoch
            # as recommended in the PyTorch documentation
            if self.scheduler is not None:
                self.scheduler.step()

            # two conditions to save a model : either saving at each epoch is requested or it is the last epoch and
            # save path was specified in the constructor method (which indicates that the user wants to save the model)
            if save_epo_state or (epoch == n_epochs - 1 and self.save_path is not None):
                self.save_state(get_dataset_name(train_loader), epoch, desc=desc, verbose=True)

            misc_print = f"lr : {self.scheduler.get_last_lr()[0]}" if self.scheduler is not None else None
            self._handle_callback("on_train_epoch_end", epoch, n_epochs,
                                  epoch_loss=running_epoch["train_loss"][0],
                                  running_val_loss=running_epoch["val_loss"][0],
                                  misc=misc_print)
            running_epoch["epoch_time"] = time.time() - epo_start_time
            df_epochs.append(running_epoch)

        out = pd.concat(df_epochs, axis=0, ignore_index=True)
        self._handle_callback("on_train_end", out)

        # saving the loss and accuracy per epoch as a csv file
        if self.save_path is not None:
            fname = self.csv_fname_template.format(
                model=self.model.__class__.__name__,
                desc=f"_{desc}" if desc is not None else "",
                optim=self.optimizer.__class__.__name__,
                dataset=get_dataset_name(train_loader),
            )
            out.to_csv(self.save_path / fname, index=False)

        return out

    def train_epoch(self, train_loader: torch_data.DataLoader, epoch_idx: int = None) -> pd.DataFrame:
        """ Perform the training for one epoch """
        torch.set_grad_enabled(True)
        self.model.train()

        running_loss, accuracy = 0, 0

        for x, y in train_loader:
            self._handle_callback("on_train_batch_start")

            self.optimizer.zero_grad()

            x, y = x.to(self.device), y.to(self.device)
            outputs = self.model(x)
            loss = self.criterion(outputs, y)

            # backward & update parameters
            loss.backward()
            self.optimizer.step()

            y_pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            accuracy += y_pred.eq(y.view_as(y_pred)).sum().item()

            running_loss += loss.item()

            self._handle_callback("on_train_batch_end")

        return pd.DataFrame({
            "epoch": [epoch_idx + 1 if epoch_idx is not None else np.nan],
            "train_loss": [running_loss / len(train_loader)],
            "train_accuracy": [accuracy / len(train_loader.dataset)]
        })

    def validation_step(self, validation_loader: torch_data.DataLoader) -> tuple[float, float]:
        """ Evaluate the model on the validation step if the validation loader is not None. This is usually done
        at the end of each epoch. """
        val_res = pd.DataFrame({"val_loss": np.nan, "val_accuracy": np.nan}, index=[0])
        if validation_loader is not None:
            val_loss, correct = self.run_test(validation_loader)
            val_res["val_loss"] = val_loss
            val_res["val_accuracy"] = correct

        return val_res

    def run_test(self, test_loader: torch_data.DataLoader) -> tuple[float, float]:
        """ Run the model on the test set and return the average loss and the accuracy. The model is set to eval
        mode. """
        self.model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                y_pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += y_pred.eq(target.view_as(y_pred)).sum().item()

        test_loss /= len(test_loader)
        correct = int(correct) / len(test_loader.dataset)

        return test_loss, correct

    def run_test_per_class(self, test_loader: torch_data.DataLoader) -> pd.DataFrame:
        """ Run the model on the test set and return the average loss and accuracy per class.

        This method set the model to the eval mode.

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            Test data loader

        Returns
        -------
        pd.DataFrame
            A DataFrame with the classes, accuracy, cross-entropy loss and the number of instances per class
        """
        self.model.eval()

        classes = get_classes_labels(test_loader.dataset)
        # classes = test_loader.dataset.classes
        predictions = pd.DataFrame({
            "classes": classes,
            "accuracy": np.zeros(len(classes)),
            "loss": np.zeros(len(classes)),
            "n_instances": np.zeros(len(classes))
        })
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        total_loss = {classname: 0 for classname in classes}

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                y_pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                for label, prediction in zip(target, y_pred):
                    if prediction == label:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1
                    total_loss[classes[label]] += self.criterion(output, target).item()

        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            predictions.loc[predictions.classes == classname, "accuracy"] = accuracy
            predictions.loc[predictions.classes == classname, "loss"] = total_loss[classname] / total_pred[classname]
            predictions.loc[predictions.classes == classname, "n_instances"] = total_pred[classname]

        return predictions

    def save_state(self, dataset_name: str, epoch: int, desc=None, verbose=False) -> None:
        # TODO add a parameter to specify the directory where to save the model
        # TODO add a identifier on the model training: epoch, dataset, major hyperparameters, etc.
        # TODO add epochs and loss / criterion in the saved data
        fname = self.model_fname_template.format(
            model=self.model.__class__.__name__,
            desc=f"_{desc}" if desc is not None else "",
            optim=self.optimizer.__class__.__name__,
            dataset=dataset_name,
            epo=epoch + 1
        )
        torch.save({
            'dataset_name': dataset_name,
            'epoch': epoch,
            'model_name': self.model.__class__.__name__,
            'optimizer_name': self.optimizer.__class__.__name__,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.save_path / fname)
        if verbose:
            self._handle_callback("on_save_state", f"Writing model state at {self.save_path / fname}")

    @classmethod
    def load_state(cls, load_path: str | pathlib.Path, fname: str, model: nn.Module, optimizer: optim.Optimizer,
                   return_trainer=False, **trainer_kwargs) -> Trainer | tuple[nn.Module, optim.Optimizer, dict]:
        """ Load the model and optimizer states from a saved state. The classes of the model and the optimizer should
        match the ones used to save the model state.

        Parameters
        ----------
        load_path : str | pathlib.Path
            The path to the directory where the model state is saved
        fname : str
            Filename of the saved model state
        model : torch.nn.Module
            The model to load the state into (should be of the same class as the saved model)
        optimizer : torch.optim.Optimizer
            The optimizer to load the state into (should be of the same class as the saved optimizer)
        return_trainer : bool
            If True, return a Trainer object with the loaded model and optimizer. If False, return the model, the
            optimizer and the checkpoint dictionary. Default is False.
        trainer_kwargs : dict
            Additional arguments to pass to the Trainer constructor. Used only if return_trainer is True.

        Returns
        -------
        Trainer | tuple[nn.Module, optim.Optimizer, dict]
            If return_trainer is True, return a Trainer object with the loaded model and optimizer. If False, return
            a tuple of the model, the optimizer and the checkpoint dictionary.
        """

        # ensure the path is a Pathlib object
        if isinstance(load_path, str):
            load_path = pathlib.Path(load_path)
        if not isinstance(load_path, pathlib.Path):
            raise ValueError("load_path should be a string or a pathlib.Path object.")

        checkpoint = torch.load(load_path / fname, mmap=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # cleaning the checkpoint dictionary
        del checkpoint['model_state_dict']
        del checkpoint['optimizer_state_dict']

        if return_trainer:
            return cls(model, optimizer, **trainer_kwargs)

        return model, optimizer, checkpoint

    def _handle_callback(self, callback_name: str, *args, **kwargs):
        """ Call the callback methods if they are defined in the CallBacks object. Those methods provide a way to
        log messages during specific stages of the training.

        Parameters
        ----------
        callback_name : str
            Name of the method to call
        args : list
            Positional arguments to pass to the callback method
        kwargs : dict
            Keyword arguments to pass to the callback method
        """
        if callback_name not in ALLOWED_CALLBACKS:
            raise ValueError(f"Callback '{callback_name}' not allowed. Allowed methods are: {ALLOWED_CALLBACKS}")

        if self.callbacks is not None and hasattr(self.callbacks, callback_name):
            getattr(self.callbacks, callback_name)(*args, **kwargs)
