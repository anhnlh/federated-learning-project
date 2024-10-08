"""
Defines the neural network model for the Federated Learning simulation.

Author: Ananya Misra, am4063@g.rit.edu
"""
#Imports functions of Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class FemnistModel(nn.Module):
    def __init__(self):
        #Calls the parent constructor 
        super(FemnistModel, self).__init__()

         # First convolutional layer: 1 input channel, 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
         # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
         # First fully connected layer: 64 * 7 * 7 input features, 128 output feature
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        
        # Output layer: 128 input features, 62 output features 
        self.fc2 = nn.Linear(in_features=128, out_features=62)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

def get_model(device):
    model = FemnistModel()
    return model.to(device)