
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn 
from tqdm import tqdm
import plotly.express as px
import pandas as pd

#Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
    
# test function
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

#load data for train and validation and test
def load_data():
    """Load MNIST (training, validation, and test set)."""
    trf = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    dataset = MNIST("./data", train=True, download=True, transform=trf)

    # Splitting dataset into train and validation sets
    #train_indices, valid_indices = train_test_split(
    #    list(range(len(dataset))), train_size=0.8,test_size=0.2, random_state=42)
    trainset,validset=train_test_split(dataset, train_size=0.8,test_size=0.2, random_state=42)
    print(len(trainset))
    print(len(validset))
    #train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    #valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(trainset, batch_size=32,shuffle=True)
    valid_loader = DataLoader(validset, batch_size=32,shuffle=True)
    print(len(train_loader))
    print(len(valid_loader))
    testset = MNIST("./data", train=False, download=True, transform=trf)
    test_loader = DataLoader(testset, batch_size=32, shuffle=True)
    print(len(test_loader))
    return train_loader, valid_loader, test_loader

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
