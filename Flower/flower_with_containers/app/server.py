import flwr as fl
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple
from flwr.common import Metrics

def getData(dist, x, y):
    dx = []
    dy = []
    counts = [0 for i in range(10)]
    for i in range(len(x)):
        if counts[y[i]]<dist[y[i]]:
            dx.append(x[i])
            dy.append(y[i])
            counts[y[i]] += 1
        
    return np.array(dx), np.array(dy)



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])




def get_evaluate_fn(model: model): #centralized evalutaion
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself


    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test[..., np.newaxis]/255.0

    # The `evaluate` function will be called after every round
    def evaluate(self, parameters, config):
        # Update model with the latest parameters
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("======================================================")
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
fl.server.start_server(server_address = "[::]:22222",config=fl.server.ServerConfig(num_rounds=12),grpc_max_message_length = 1024*1024*1024, strategy = strategy)
