
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn 
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
precision = 2**64
threshold = 10

def encrypt(val, pk, coef=1):
    """
      To deal with negative values, we add threshold to all values.
      if value+threshold is still negative we replace it by 0
    """
    if isinstance(val, np.ndarray):
        if val.ndim == 1:
            ret = np.array([str(int(pk.encrypt(int((value.item()+threshold)*precision*coef)).ciphertext())) for value in val], dtype=str)
            return ret
        else:
            return np.array([encrypt(sub_val, pk, coef) for sub_val in val], dtype=str)
    elif isinstance(val, list):
        return [encrypt(sub_val, pk, coef) for sub_val in val]
    else:
        return str(int(pk.encrypt(int((val+threshold)*precision*coef)).ciphertext()))

def decrypt(val, sk):
    if isinstance(val, np.ndarray):
        if val.ndim == 1:
            dec = np.array([sk.raw_decrypt(int(value))/precision -threshold for value in val]).astype(np.float64)
            return dec
        else:
            return np.array([decrypt(sub_val, sk) for sub_val in val])
    elif isinstance(val, list):
        return [decrypt(sub_val, sk) for sub_val in val]
    else:
        return sk.raw_decrypt(int(val))/precision - threshold
    
    
def print_first_value(param_array, transformation=False):
    params_print = param_array.flatten()
    print(params_print[0])
    if transformation:
        print(int((params_print[0] + threshold)*precision))    


#Model definition
class Net(nn.Module):
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.fc = nn.Linear(784, 10)

    # def forward(self, x):
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)
    #     return x

    
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.flatten = nn.Flatten()
    #     self.fc1 = nn.Linear(28*28, 128)
    #     self.relu1 = nn.ReLU()
    #     self.fc2 = nn.Linear(128, 256)
    #     self.relu2 = nn.ReLU()
    #     self.fc3 = nn.Linear(256, 10)
    #     self.softmax = nn.Softmax(dim=1)

    # def forward(self, x):
    #     x = self.flatten(x)
    #     x = self.fc1(x)
    #     x = self.relu1(x)
    #     x = self.fc2(x)
    #     x = self.relu2(x)
    #     x = self.fc3(x)
    #     x = self.softmax(x)
    #     return x


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
    
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.flatten = nn.Flatten()
    #     self.linear_relu_stack = nn.Sequential(
    #         nn.Linear(784, 128),
    #         nn.ReLU(),
    #         nn.Linear(128, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, 10)
    #         ## Softmax layer ignored since the loss function defined is nn.CrossEntropy()
    #     )

    # def forward(self, x):
    #     x = self.flatten(x)
    #     logits = self.linear_relu_stack(x)
    #     return  logits
    
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
    trainset,validset=train_test_split(dataset, train_size=0.8,test_size=0.2, random_state=42)


    train_loader = DataLoader(trainset, batch_size=32,shuffle=True)
    valid_loader = DataLoader(validset, batch_size=32,shuffle=True)

    testset = MNIST("./data", train=False, download=True, transform=trf)
    test_loader = DataLoader(testset, batch_size=32, shuffle=True)

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
