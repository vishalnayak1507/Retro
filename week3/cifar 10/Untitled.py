#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# Define transformation pipeline for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image or numpy.ndarray to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1] range
])

batch_size = 32  # Reduce batch size due to memory issues

# Load CIFAR-10 dataset and create data loaders
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Define classes in CIFAR-10 dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[5]:


# Function to display images
def display_images(images, labels):
    for j in range(len(images)):
        plt.figure(figsize=(5, 5))
        image = images[j] / 2 + 0.5  # Unnormalize
        npimg = image.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(f'Ground Truth: {classes[labels[j]]}')
        plt.axis('off')
        plt.show()


# In[6]:


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels (RGB), 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with 2x2 kernel and stride
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 input channels, 16 output channels, 5x5 kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Fully connected layer: 16 * 5 * 5 input features, 120 output features
        self.fc2 = nn.Linear(120, 84)  # Fully connected layer: 120 input features, 84 output features
        self.fc3 = nn.Linear(84, 10)  # Fully connected layer: 84 input features, 10 output features (classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))  # Convolution -> ReLU -> Max pooling
        x = self.pool(nn.functional.relu(self.conv2(x)))  # Convolution -> ReLU -> Max pooling
        x = torch.flatten(x, 1)  # Flatten 2D features to 1D
        x = nn.functional.relu(self.fc1(x))  # Fully connected -> ReLU
        x = nn.functional.relu(self.fc2(x))  # Fully connected -> ReLU
        x = self.fc3(x)  # Output layer
        return x

# Create an instance of the CNN model
model = CNN()


# In[7]:


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross entropy loss function for multi-class classification
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # SGD optimizer with learning rate and momentum


# In[8]:


# Training loop
epochs = 10  # Number of training epochs
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data  # Get the inputs
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize
        running_loss += loss.item()  # Accumulate loss
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')


# In[10]:


# Test the model on the test dataset
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct / total} %')


# In[12]:


# Calculate accuracy per class
class_correct = list(0. for _ in range(10))
class_total = list(0. for _ in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):  # Iterate over actual batch size
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# Print accuracy for each class
for i in range(10):
    if class_total[i] > 0:
        print(f'Accuracy of {classes[i]} : {100 * class_correct[i] / class_total[i]} %')
    else:
        print(f'Accuracy of {classes[i]} : N/A (no samples)')


# In[ ]:




