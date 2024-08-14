import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import itertools
from torchtext.datasets import IMDB
from collections import Counter

'''In PyTorch, handling this sequential nature typically involves tokenization, embedding layers, 
and the use of recurrent layers or attention mechanisms to process the data while preserving the 
order of the words.'''


# Download the MNIST dataset
transform = transforms.ToTensor()
train_dataset, test_dataset = torchtext.datasets.IMDB(root: str = '.data', split: Union[Tuple[str], str] = ('train', 'test'))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
num_classes = 10

# Encoding the text into tensors (one-hot is the most common and simple way)


def tokenize(text):
    return text.lower().split()

train_iter, test_iter = IMDB()

counter = Counter()

for label, text in itertools.chain(train_iter, test_iter):
    tokenized_text = tokenize(text)
    counter.update(tokenized_text)


# Plot the loss curve
plt.figure(figsize=(8, 6))
plt.plot(range(num_epochs), train_losses, label="Training loss")
plt.plot(range(num_epochs), val_losses, label="Validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.show()
