
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn 
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import torch.nn.functional as F


#Models definition
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class Net(nn.Module):  
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(     

            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=2, stride=2)   
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),    
            nn.MaxPool2d(2, 2)  
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
class CNN(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


#Test function for clients and server 
def test(net, testloader, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss=loss/len(testloader.dataset)
    return loss, accuracy



#Train fuction used by clients
def train(net, trainloader,  device: str, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()



#Plot losses and accuracies curves
def plot_loss_accuracy(losses, accuracies):
    data = {'Round': range(1, len(losses) + 1),
            'Loss': losses,
            'Accuracy': accuracies}

    # Create DataFrame
    df = pd.DataFrame(data)

    # Plot loss curve
    loss_plot = px.line(df, x='Round', y='Loss', markers=True, title="Loss")

    # Plot accuracy curve
    accuracy_plot = px.line(df, x='Round', y='Accuracy', markers=True, title="Accuracy")

    return (loss_plot,accuracy_plot)
