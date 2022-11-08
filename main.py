import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        torch.nn.Conv2d(75 , (3,3) , strides = 1 ,input_shape = (28,28,1))
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

