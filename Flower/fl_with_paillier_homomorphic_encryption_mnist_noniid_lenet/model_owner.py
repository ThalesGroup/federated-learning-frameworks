
import numpy as np
from phe import paillier
import pickle
import torch.nn as nn 
import sys

precision = 2**64
threshold = 10

def KeyGen(n_length):
    public_key, private_key = paillier.generate_paillier_keypair(n_length=n_length)
    with open('public_key.pkl', 'wb') as f:
        pickle.dump(public_key, f)
    with open('private_key.pkl', 'wb') as f:
        pickle.dump(private_key, f)


def decrypt(val):
    with open('/my_app/private_key.pkl', 'rb') as f:
        sk = pickle.load(f)
    if isinstance(val, np.ndarray):
        if val.ndim == 1:
            dec = np.array([sk.raw_decrypt(int(value))/precision -threshold for value in val]).astype(np.float64)
            return dec
        else:
            return np.array([decrypt(sub_val) for sub_val in val])
    elif isinstance(val, list):
        return [decrypt(sub_val) for sub_val in val]
    else:
        return sk.raw_decrypt(int(val))/precision - threshold

#Model definition
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
    


if __name__ == '__main__':
    n_length = 128
    if len(sys.argv)>=2:
        n_length=sys.argv[1]
    KeyGen(n_length=n_length)
    
