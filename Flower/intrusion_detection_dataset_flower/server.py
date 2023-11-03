import flwr as fl
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple
from flwr.common import Metrics

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


# Load train dataset
train = pd.read_csv("./my_data_result/train_data.csv")
X_train_scaled = scaler.fit_transform(train.drop(['y'], axis=1).to_numpy())
y_train = train['y'].to_numpy()


# Load Test dataset
test=pd.read_csv("./my_data_result/test_data.csv")
X_test_scaled = scaler.transform(test.drop(['y'], axis=1).to_numpy())
y_test = test['y'].to_numpy()
y_test_cat = to_categorical(y_test)

model=model_def(X_test_scaled.shape[1],y_test_cat.shape[1])




def get_evaluate_fn(model: model): #centralized evalutaion
    """Return an evaluation function for server-side evaluation."""

    # The `evaluate` function will be called after every round
    def evaluate(self, parameters, config):
        # Update model with the latest parameters
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
        return {"Test loss:": loss}, {"Test accuracy:": accuracy}
    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:  #distributed evaluation
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Create strategy and run server
strategy = fl.server.strategy.FedAvg(min_available_clients=2,evaluate_metrics_aggregation_fn=weighted_average, evaluate_fn=get_evaluate_fn(model)
                                     )
# Start Flower server for three rounds of federated learning
fl.server.start_server(server_address = "localhost:8081",
                       config=fl.server.ServerConfig(num_rounds=10),
                       grpc_max_message_length = 1024*1024*1024, 
                       strategy = strategy)
