
from collections import OrderedDict
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import utils 



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = utils.CNN().to(DEVICE)
losses = []
accuracies = []



#Data 
trf =Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = CIFAR10("./data", train=True, download=True, transform=trf)
dist_client1 = [500, 110, 7, 690, 7000, 5100, 4000, 4200, 113, 5000]
client1_indices = []
for class_label in range(10):
    class_indices = [i for i, (_, label) in enumerate(train_dataset) if label == class_label]
    client1_count = dist_client1[class_label]
    client1_indices.extend(class_indices[:client1_count])
client1_dataset = Subset(train_dataset, client1_indices)


trainset,validset=train_test_split(client1_dataset, train_size=0.8,test_size=0.2, random_state=42)
print(len(trainset))
print(len(validset))

trainloader = DataLoader(trainset, batch_size=32,shuffle=True)
validloader = DataLoader(validset, batch_size=32,shuffle=True)

print("End data_Load")

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        utils.train(net, trainloader,DEVICE, epochs=5)
        return self.get_parameters(config={}), len(trainloader.dataset), {}
    #Local test
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = utils.test(net, validloader,DEVICE)
        losses.append(loss)
        accuracies.append(accuracy)
        print("Validation loss",loss,"  Validation accuracy:",accuracy)
        return loss, len(validloader.dataset), {"accuracy": accuracy}

# Start Flower client
fl.client.start_client(
    server_address="localhost:8081",
    client=FlowerClient().to_client(),
    grpc_max_message_length=1024*1024*1024
)

print("client2 losses:")
print(losses)
print("client2 accuracies")
print(accuracies)
loss_plot,accuracy_plot=utils.plot_loss_accuracy(losses, accuracies)
loss_plot.write_image("client2_validation_losses.png")
accuracy_plot.write_image("client2_validation_accuracies.png")
