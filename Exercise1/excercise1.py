import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt

"""""
Make a simple neural network using PyTorch. The goal of this exercise is to look at the
MNIST dataset, and tell what numbers are written. Create a model with 3 fully connected
hidden layers, and a 10-unit output layer.
"""

# Download the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True,
transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True,
transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
batch_size=64, shuffle=False)

""""
1. Create a neural network:
    a. Initialize 3 layers
    b. Define the forward function:
        i. Reshape the data to a fully connected layer. Hint: Use .view() or .flatten().
        ii. Let the input pass through the different layers.
        iii. Consider what activation function you want to use in between the layers, and for the final layer.
    c. Loss function and optimizer:
        i. Consider what loss function and optimizer you want to use.
    d. Create the training loop:
    e. Create the evaluation loop:
    f. Save the model
2. Report your accuracy, is this satisfactory? Why / why not?
3. Plot the loss curve.
"""


