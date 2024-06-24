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
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range (mean=0.5, std=0.5 for grayscale)
])

batch_size = 64

# Load MNIST dataset and create data loaders
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Define classes in MNIST dataset
classes = tuple(str(i) for i in range(10))


# In[3]:


# Function to display images
def display_images(images, labels):
    for j in range(len(images)):
        plt.figure(figsize=(3, 3))
        image = images[j] / 2 + 0.5  # Unnormalize
        npimg = image.squeeze().numpy()
        plt.imshow(npimg, cmap='gray')
        plt.title(f'Ground Truth: {classes[labels[j]]}')
        plt.axis('off')
        plt.show()


# In[4]:


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel (grayscale), 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with 2x2 kernel and stride
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 input channels, 16 output channels, 5x5 kernel
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Fully connected layer: 16 * 4 * 4 input features, 120 output features
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
cnn_model = CNN()


# In[5]:


# Define the Feed-forward Neural Network (FNN) model
class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Fully connected layer: 28 * 28 input features, 128 output features
        self.fc2 = nn.Linear(128, 64)  # Fully connected layer: 128 input features, 64 output features
        self.fc3 = nn.Linear(64, 10)  # Fully connected layer: 64 input features, 10 output features (classes)

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten 2D features to 1D
        x = nn.functional.relu(self.fc1(x))  # Fully connected -> ReLU
        x = nn.functional.relu(self.fc2(x))  # Fully connected -> ReLU
        x = self.fc3(x)  # Output layer
        return x

# Create an instance of the FNN model
fnn_model = FNN()


# In[6]:


# Define loss function and optimizer for CNN
cnn_criterion = nn.CrossEntropyLoss()  # Cross entropy loss function for multi-class classification
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)  # Adam optimizer with learning rate

# Define loss function and optimizer for FNN
fnn_criterion = nn.CrossEntropyLoss()  # Cross entropy loss function for multi-class classification
fnn_optimizer = optim.Adam(fnn_model.parameters(), lr=0.001)  # Adam optimizer with learning rate


# In[7]:


# Training loop for CNN
def train_cnn(model, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'CNN [{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training CNN')

# Train the CNN model
train_cnn(cnn_model, cnn_criterion, cnn_optimizer, epochs=5)


# In[8]:


# Test and evaluate accuracy for CNN
def test_cnn(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'CNN Accuracy on the test dataset: {100 * correct / total:.2f} %')

# Test CNN model
test_cnn(cnn_model)


# In[9]:


# Test and evaluate accuracy for FNN
def test_fnn(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.view(images.size(0), -1)  # Flatten input for FNN
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'FNN Accuracy on the test dataset: {100 * correct / total:.2f} %')

# Test FNN model
test_fnn(fnn_model)


# In[ ]:




