"""
Defines the neural network model for the Federated Learning simulation.

Author: Ananya Misra, am4063@g.rit.edu; Anh Nguyen, aln4739@rit.edu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FemnistModel(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):  # Keep original num_classes
        super(FemnistModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=32,
            kernel_size=3,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(
            64 * ((input_shape[1] // 4 - 2) * (input_shape[2] // 4 - 2)), num_classes
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)  # Keep original softmax


def get_model(device):
    model = FemnistModel()
    return model.to(device)
