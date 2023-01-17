import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torchvision.datasets import MNIST


def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
    transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x))
        ])

    train_loader = DataLoader(
        MNIST('./data/', train=True, download=True, transform=transform),
        batch_size=train_batch_size, 
        shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False, download=True, transform=transform),
        batch_size=test_batch_size, 
        shuffle=False)

    return train_loader, test_loader



def merge_label(x, y):
    x_ = x.clone()
    x_[:, :10] *= 0.0  # first 10 items of each row get zeroed (for every item in batch)
    x_[range(x.shape[0]), y] = x.max()
    return x_


def loss_fn(g_pos, g_neg, threshold):
    return torch.log(1 + torch.exp(torch.cat([-g_pos + threshold, g_neg - threshold]))).mean()


def goodness(x):
    return x.pow(2).mean(1)


class LPNormalise(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.functional.normalize

    def forward(self, x):
        return self.norm(x)