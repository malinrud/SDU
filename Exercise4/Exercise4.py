import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt

# Download the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
num_classes = 10

# Dataset processing
train_dimensions = train_dataset.data.dim
test_dimensions = test_dataset.data.dim

print(f"Train dataset size: {train_dataset.data.shape}")
print(f"Test dataset size: {test_dataset.data.shape}")
print(f"Image size: {train_dataset.data[0].shape}") # this is the image

# Counting occurences of number and printing
plt.bar(range(10), train_dataset.targets.bincount().numpy())
plt.title('Digit distribution in MNIST dataset')
plt.xlabel('Digit')
plt.ylabel('Count')
plt.show()

# Design Convolutionary Neural Network
class NNetwork(nn.Module):
    def __init__(self):
        super(NNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3) #Convolutional Layer
        self.maxpool1 = nn.MaxPool2d(kernel_size=2 )#MaxPool2D Layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3) #Fully Connected Layer
        self.maxpool2 = nn.MaxPool2d(kernel_size=2 )#MaxPool2D Layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3) #Fully Connected Layer
        
        
        self.fc1 = nn.Linear(10*3*3, 128)  
        self.fc2 = nn.Linear(16, 16) 
        self.dropout = nn.Dropout(0.2) # Dropout Layer 
        self.relu = nn.ReLU #ReLU Activation Function

    def forward(self, x):
        x = x.flatten(start_dim=1)  # input layer    
        x = torch.relu(self.fc1(x)) # first layer  
        x = torch.relu(self.fc2(x)) # second layer   
        x = self.fc3(x) # Third layer
        # (this applies to self.fc3(x))) No softmax because we use CrossEntropyLoss
        # When using the softmax activation along with the cross-entropy loss 
        # (which is LogSoftmax + NLLLoss combined into one function), 
        # the two functions effectively cancel each other out, leading to incorrect 
        # gradients and learning behavior.     
        return x
    
model = NNetwork()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) # without momentum to show learning curve. You get better performance with momentum

train_losses = []
val_losses = []

num_epochs = 50
for epoch in range(num_epochs):
    model.train() 
    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad() # Clear previous gradients
        outputs = model(data) # Forward pass
        loss = criterion(outputs, targets) # Compute loss
        loss.backward() # Backward pass (compute gradients)
        optimizer.step() # Update weights
        
    train_losses.append(loss.item()) # Record the training loss for this epoch

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # 1b Report your accuracy, is this satisfactory? 
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    val_losses.append(loss.item())


torch.save(model.state_dict(), 'model.pth')

# 1c Plot the loss curve.
plt.figure(figsize=(8, 6))
plt.plot(range(num_epochs), train_losses, label="Training loss")
plt.plot(range(num_epochs), val_losses, label="Validation loss")
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('graph.png')
plt.show()
