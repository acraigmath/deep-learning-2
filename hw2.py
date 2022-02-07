# Adam Craig
# Deep Learning
# LaNet-style CNN on CIFAR-10
# HW2

# Note: Much of this code is structurally built off of much of the code found
# on the PyTorch official tutorials, so
# credit to those people kind enough to create their tutorial

import sys
from typing import List
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        X, y = data
        y = F.one_hot(y, num_classes=10)
        y = y.float()
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            y = F.one_hot(y, num_classes=10)
            y = y.float()
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def test_network(testing_data, training_data, flag, lr_list: List, batch_list: List, activation_list: List, epochs: int, loss_list: List):
    for learning_rate in lr_list:
        for batch_size in batch_list:
            for activation_function in activation_list:
                for loss in loss_list:
                    print(f"Iteration: (LR, A, L) = ({learning_rate},{str(activation_function)}, {str(loss)})")
                        
                    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
                    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

                    #create model architecture
                    if flag == 1:
                        class CNN(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
                                self.pool = nn.AvgPool2d(2, 2)
                                self.conv2 = nn.Conv2d(6, 16, 5)
                                self.fc1 = nn.Linear(16 * 6 * 6, 120)
                                self.fc2 = nn.Linear(120, 84)
                                self.fc3 = nn.Linear(84, 10)

                            def forward(self, x):
                                x = self.pool(activation_function(self.conv1(x)))
                                x = self.pool(activation_function(self.conv2(x)))
                                x = torch.flatten(x, 1) # flatten all dimensions except batch
                                x = activation_function(self.fc1(x))
                                x = activation_function(self.fc2(x))
                                x = self.fc3(x)
                                return x
                    if flag == 2:
                        class CNN(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.conv1 = nn.Conv2d(3, 6, 3, padding=2)
                                self.pool = nn.AvgPool2d(2, 2)
                                self.conv2 = nn.Conv2d(6, 16, 3)
                                self.fc1 = nn.Linear(16 * 7 * 7, 120)
                                self.fc2 = nn.Linear(120, 84)
                                self.fc3 = nn.Linear(84, 10)

                            def forward(self, x):
                                x = self.pool(activation_function(self.conv1(x)))
                                x = self.pool(activation_function(self.conv2(x)))
                                x = torch.flatten(x, 1) # flatten all dimensions except batch
                                x = activation_function(self.fc1(x))
                                x = activation_function(self.fc2(x))
                                x = self.fc3(x)
                                return x
                    if flag == 3:
                        class CNN(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.pool = nn.AvgPool2d(2, 2)

                                self.conv1 = nn.Conv2d(3, 6, 3, padding=2)
                                self.conv2 = nn.Conv2d(6, 12, 3, padding=2)
                                self.conv3 = nn.Conv2d(12, 24, 3, padding=2)
                                self.conv4 = nn.Conv2d(24, 48, 3, padding=2)
                                self.conv5 = nn.Conv2d(48, 96, 3, padding=2)

                                self.fc1 = nn.Linear(96 * 2 * 2, 120)
                                self.fc2 = nn.Linear(120, 84)
                                self.fc3 = nn.Linear(84, 10)

                            def forward(self, x):
                                x = self.pool(activation_function(self.conv1(x)))
                                x = self.pool(activation_function(self.conv2(x)))
                                x = self.pool(activation_function(self.conv3(x)))
                                x = self.pool(activation_function(self.conv4(x)))
                                x = self.pool(activation_function(self.conv5(x)))
                                x = torch.flatten(x, 1) # flatten all dimensions except batch
                                x = activation_function(self.fc1(x))
                                x = activation_function(self.fc2(x))
                                x = self.fc3(x)
                                return x            

                    model = CNN()
                    loss_fn = loss
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                    for t in range(epochs):
                        print(f"Epoch {t+1}\n-------------------------------")
                        train_loop(train_dataloader, model, loss_fn, optimizer)
                        test_loop(test_dataloader, model, loss_fn)
                    print("Done!")

def main():
    if len(sys.argv) == 1:
        print("Please enter an argument for which part to complete. Ex: `python hw2.py 2`")
        return
    flag = int(sys.argv[1])

    # normalize image data for classification
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    testing_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    if flag == 1:
        # Part 1
        lr_list = [0.001, 0.01, 0.1]
        batch_list = [64]
        activation_list = [torch.tanh, torch.sigmoid]
        epochs = 10
        loss_list = [nn.MSELoss(), nn.CrossEntropyLoss()]
    
        test_network(testing_data, training_data, flag, lr_list=lr_list, batch_list=batch_list, activation_list=activation_list, epochs=epochs, loss_list=loss_list)

    if flag == 2:
        # Part 2
        lr_list = [0.001]
        batch_list = [64]
        activation_list = [torch.relu]
        epochs = 30
        loss_list = [nn.CrossEntropyLoss()]
    
        test_network(testing_data, training_data, flag, lr_list=lr_list, batch_list=batch_list, activation_list=activation_list, epochs=epochs, loss_list=loss_list)    

    if flag == 3:
        # Part 3
        lr_list = [0.001]
        batch_list = [64]
        activation_list = [torch.relu]
        epochs = 30
        loss_list = [nn.CrossEntropyLoss()]
    
        test_network(testing_data, training_data, flag, lr_list=lr_list, batch_list=batch_list, activation_list=activation_list, epochs=epochs, loss_list=loss_list)    

main()
