
from collections import OrderedDict
import flwr as fl
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
import utils 
import model_owner
import numpy as np
import pickle

torch.manual_seed(0)
np.random.seed(0)
precision = 2**64
threshold = 10

#Load public-secret key
with open('/my_app/public_key.pkl', 'rb') as f:
    pk = pickle.load(f)
print("Public Key",pk)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = model_owner.Net().to(DEVICE)
losses = []
accuracies = []


#NonIID Data 
trf = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
train_dataset = MNIST("./data", train=True, download=True, transform=trf)
dist_client1 = [8100, 4100, 4600, 7000, 50, 50, 10, 0, 7100, 70]
client1_indices = []
for class_label in range(10):
    class_indices = [i for i, (_, label) in enumerate(train_dataset) if label == class_label]

    client1_count = dist_client1[class_label]
    client1_indices.extend(class_indices[:client1_count])

client1_dataset = Subset(train_dataset, client1_indices)



trainset,validset=train_test_split(client1_dataset, train_size=0.8,test_size=0.2, random_state=42)

trainloader = DataLoader(trainset, batch_size=32,shuffle=True)
validloader = DataLoader(validset, batch_size=32,shuffle=True)



# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        coef=1
        if 'nb_data' in config.keys():
            coef = 1/config['nb_data']
        params=[val.cpu().numpy()for _, val in net.state_dict().items()] 
        print("get_parameters")
        print(params[0].shape)
        utils.print_first_value(params[0], transformation=True)
        enc=utils.encrypt(params,pk, coef)
        utils.print_first_value(enc[0])
        return enc

    def set_parameters(self, parameters):
        print("set parameters")
        parameters = [p.astype(object) for p in parameters]
        utils.print_first_value(parameters[0])
        parameters=model_owner.decrypt(parameters)
        utils.print_first_value(parameters[0], transformation=True)
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("fit")
        self.set_parameters(parameters)
        
        utils.train(net, trainloader,DEVICE, epochs=1)
        config['nb_data']=2
        s=self.get_parameters(config=config)

        return s, len(trainloader.dataset), {}
    #Local test
    def evaluate(self, parameters, config):
        print("evaluate")
        self.set_parameters(parameters)
        loss, accuracy = utils.test(net, validloader,DEVICE)
        losses.append(loss)
        accuracies.append(accuracy)
        print("Validation loss",loss,"  Validation accuracy:",accuracy)
        return loss, len(validloader.dataset), {"accuracy": accuracy}

# Start Flower client
fl.client.start_client(
        server_address="server:22222",
    client=FlowerClient().to_client(),
    grpc_max_message_length=1024*1024*1024
)
print("client1 losses:")
print(losses)
print("client1 accuracies")
print(accuracies)
loss_plot,accuracy_plot=utils.plot_loss_accuracy(losses, accuracies)
loss_plot.write_image("client1_validation_losses.png")
accuracy_plot.write_image("client1_validation_accuracies.png")
