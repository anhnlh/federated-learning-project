"""
Client implementation using the Flower Framework.

Author: Anh Nguyen, aln4739@rit.edu, Ananya Misra, am4063@g.rit.edu;
"""

import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from src.main import get_device, get_weights, load_data, set_weights
from src.model import get_model


class FemnistClient(NumPyClient):
    """
    Client implementation for the FEMNIST dataset.
    """

    def __init__(
            self,
            model,
            train_loader,
            test_loader,
            local_epochs,
            device,
            client_id,
            is_poisoned,
            config_string,
            folder
    ):
        """
        Initializes the client with the given model, data loaders, number of local epochs, and device.

        Args:
            model (torch.nn.Module): The neural network model to be trained.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
            local_epochs (int): Number of local epochs to train the model.
            device (torch.device): The device (CPU or GPU) on which the model will be trained.
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.device = device
        self.client_id = client_id
        self.is_poisoned = is_poisoned

        # Stuff for writing metrics to file
        self.folder = folder
        self.filename = f"{self.folder}/client_{client_id}_metrics({config_string}).csv"
        os.makedirs(self.folder, exist_ok=True)

    def fit(self, parameters, _):
        """
        Trains the model using the provided parameters and returns the updated model weights,
        the size of the training dataset, and the training results.

        Args:
            parameters (list): A list of parameters to set the model weights.
            _ (Any): Placeholder for additional arguments (not used).
        Returns:
            tuple: A tuple containing:
            - list: The updated model weights.
            - int: The size of the training dataset.
            - dict: The results of the training process.
        """
        set_weights(self.model, parameters)
        train_loss, train_accuracy = train(
            self.model,
            self.train_loader,
            self.local_epochs,
            self.device,
            self.is_poisoned,
        )
        cur_round_num = 1
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                reader = csv.DictReader(f)
                for _ in reader:
                    cur_round_num += 1

        with open(self.filename, "a+", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["round", "loss", "accuracy"]
            )
            if cur_round_num == 1:
                writer.writeheader()
            writer.writerow({"round": cur_round_num, "loss": train_loss, "accuracy": train_accuracy})
        return (
            get_weights(self.model),
            len(self.train_loader) * self.train_loader.batch_size,
            {"loss": train_loss, "accuracy": train_accuracy},
        )

    def evaluate(self, parameters, _):
        """
        Evaluate the model on the test dataset.

        Args:
            parameters (list): The model parameters to be set before evaluation.
            _ (Any): Placeholder for additional arguments (not used).
        Returns:
            tuple: A tuple containing:
            - loss (float): The loss value after evaluation.
            - dataset_size (int): The size of the test dataset.
            - metrics (dict): A dictionary containing evaluation metrics, such as accuracy.
        """
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader, self.device)
        return (
            loss,
            len(self.test_loader) * self.test_loader.batch_size,
            {"loss": loss, "accuracy": accuracy},
        )


def train(model, train_loader, epochs, device, is_poisoned):
    """
    Trains the given model using the provided training data loader for a specified number of epochs.
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader providing the training data.
        epochs (int): Number of epochs to train the model.
        device (torch.device): The device (CPU or GPU) on which to perform training.
    Returns:
        float: The average loss over the training data.
    """
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())
    model.train()
    total_loss, total_correct = 0, 0
    total_samples = 0

    for _ in range(epochs):
        for batch in train_loader:
            images = batch["img"]
            labels = batch["label"]
            if is_poisoned:
                poisoned_labels = torch.randint(0, 10, labels.shape).to(device)
                # if poisoned label is the same, add 1 to ensure the label is different
                labels = (poisoned_labels + (poisoned_labels == labels).int()) % 10
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def test(model, test_loader, device):
    """
    Evaluate the model on the test dataset.
    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): The device (CPU or GPU) on which the model is deployed.
    Returns:
        tuple: A tuple containing:
            - loss (float): The cumulative loss over the test dataset.
            - accuracy (float): The accuracy of the model on the test dataset.
    """
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()
    total_loss, total_correct = 0, 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["img"]
            labels = batch["label"]
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def client_fn(context: Context):
    """
    Initializes and returns a FemnistClient instance configured for federated learning.
    Args:
        context (Context): The context object containing node and run configurations.
    Returns:
        FemnistClient: An instance of FemnistClient configured with the specified model,
                       data loaders, local epochs, and device.
    """
    device = get_device()
    # read node_config
    client_id = context.node_config["partition-id"]

    # read run_config
    data_dir = context.run_config["data-dir"]
    batch_size = context.run_config["batch-size"]
    train_loader, test_loader = load_data(client_id, data_dir, batch_size, device)
    local_epochs = context.run_config["local-epochs"]
    unlucky = [1, 2, 5]
    is_poisoned = context.run_config["poison"] and client_id in unlucky
    num_rounds = context.run_config["num-server-rounds"]

    config_string = f"num_rounds={num_rounds}, local_epochs={local_epochs}, batch_size={batch_size}"

    return FemnistClient(
        model=get_model(device),
        train_loader=train_loader,
        test_loader=test_loader,
        local_epochs=local_epochs,
        device=device,
        client_id=client_id,
        is_poisoned=is_poisoned,
        config_string=config_string,
        folder="results/attack" if context.run_config["poison"] else "results/no_attack",
    ).to_client()


app = ClientApp(client_fn)
