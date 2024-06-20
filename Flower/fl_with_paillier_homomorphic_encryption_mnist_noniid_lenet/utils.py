from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
import torch
from sklearn.model_selection import train_test_split
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

    
def print_first_value(param_array, transformation=False):
    params_print = param_array.flatten()
    print(params_print[0])
    if transformation:
        print(int((params_print[0] + threshold)*precision))    
 
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
