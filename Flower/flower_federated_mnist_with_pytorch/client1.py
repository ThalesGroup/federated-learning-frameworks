
from collections import OrderedDict
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import utils 


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = utils.Net().to(DEVICE)
trainloader, validloader, testloader = utils.load_data()
losses = []
accuracies = []

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

loss_plot,accuracy_plot=utils.plot_loss_accuracy(losses, accuracies)
loss_plot.write_image("client2_validation_losses.png")
accuracy_plot.write_image("client2_validation_accuracies.png")
