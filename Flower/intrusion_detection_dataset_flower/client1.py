import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
scaler=MinMaxScaler()
from Model import model_def



# Load dataset
train = pd.read_csv("./my_data_result/train_data.csv")
X_train_scaled = scaler.fit_transform(train.drop(['y'], axis=1).to_numpy())
y_train = train['y'].to_numpy()
y_train_cat = to_categorical(y_train)
x_trn, x_vld, y_trn, y_vld = train_test_split(X_train_scaled, y_train_cat,train_size=0.8, test_size=0.2, random_state=42)


model=model_def(X_train_scaled.shape[1],y_train_cat.shape[1])



# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(x_trn, y_trn, epochs=5
                      , validation_data=(x_vld, y_vld)
                      , verbose=0)
        hist = r.history
        print("Fit history : " ,hist)
        return model.get_weights(), len(x_trn), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_vld, y_vld, verbose=0)
        print("Validation loss:",loss,"  Validation accuracy:", accuracy)
        return loss, len(x_vld), {"accuracy" :accuracy}

# Start Flower client
fl.client.start_client(server_address="localhost:8081", client=FlowerClient(.to_client(), grpc_max_message_length = 1024*1024*1024)
