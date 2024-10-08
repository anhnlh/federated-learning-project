import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from model import FemnistModel

class FemnistClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    #def get_parameters():
        

    #def set_parameters():
      

    #def fit():
       

    #def evaluate():
       

#def train():
   
#def test():
    