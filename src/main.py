"""
Main script to run the Federated Learning simulation.

Author: Anh Nguyen, aln4739@rit.edu
"""

import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor


class ClientDataset(Dataset):
    """
    Custom Pytorch Dataset class for each client.

    Args:
        image_paths (list): List of image paths.
        labels (list): List of labels corresponding to the image paths.
    """
    def __init__(self, image_paths, labels, device):
        self.image_paths = image_paths
        self.labels = labels
        self.device = device
        self.transform = Compose([
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]
        image = self.transform(image).to(self.device)
        label = torch.tensor(label).to(self.device)

        return {'img': image, 'label': label}


def split_dataset(client_data_dir):
    """
    Split the dataset into training and testing sets.
    :param client_data_dir: path to the client data directory.
    :return: train and test sets.
    """
    image_paths = []
    labels = []

    for label in os.listdir(client_data_dir):
        label_dir = os.path.join(client_data_dir, label)
        for img_file in os.listdir(label_dir):
            image_paths.append(os.path.join(label_dir, img_file))
            labels.append(int(label))

    train_size = int(0.8 * len(image_paths))
    train_image_paths = image_paths[:train_size]
    train_labels = labels[:train_size]
    test_image_paths = image_paths[train_size:]
    test_labels = labels[train_size:]

    return train_image_paths, train_labels, test_image_paths, test_labels


def visualize_data(client_id, train_loader):
    """
    Visualize the data for each client.
    :param client_id: client id.
    :param train_loader: train data loader.
    """
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle(f"Client {client_id} Data", fontsize=16, y=0.95)
    for batch in train_loader:
        for i in range(5):
            img, label = batch['img'][i], batch['label'][i]
            img = img.permute(1, 2, 0)
            img = img * 0.5 + 0.5
            axes[i].imshow(img)
            axes[i].set_title(f"Label: {label}", y=-0.2)
            axes[i].axis('off')
        break
    plt.show()

def set_device():
    """
    Set the device to the first available GPU, otherwise use CPU.
    :return: device.
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            if not torch.cuda.get_device_properties(i).is_available:
                continue
            return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def main():
    """
    Main function to run the Federated Learning simulation.
    """
    device = set_device()
    data_dir = './femnist_subset'
    num_clients = len(os.listdir(data_dir))

    for client_id in range(num_clients):
        train_image_paths, train_labels, test_image_paths, test_labels = split_dataset(
            os.path.join(data_dir, f"client_{client_id}")
        )
        train_dataset = ClientDataset(train_image_paths, train_labels, device)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataset = ClientDataset(test_image_paths, test_labels, device)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        visualize_data(client_id, train_loader)  # comment out if no visualization
        # TODO: initialize ClientApp


if __name__ == '__main__':
    main()
