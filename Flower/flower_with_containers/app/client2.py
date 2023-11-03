import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split



# AUxillary methods
def getDist(y_train):
    y_train_labels=[str(label) for label in y_train]
    ax=sns.countplot(x=y_train_labels,hue=y_train)
    ax.set(title="Count of data classes")
    plt.show()


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

# Load and compile Keras model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
dist = [0, 10, 10, 10, 4000, 3000, 4000, 5000, 10, 4500]
x_train, y_train = getData(dist, x_train, y_train)
getDist(y_train)

x_trn, x_vld, y_trn, y_vld = train_test_split(x_train, y_train,train_size=0.8, test_size=0.2, random_state=42)

# n_trn=int(0.8*len(x_train))
# n_vld=int(0.1*len(x_train))

# x_trn,y_trn=x_train[:n_trn],y_train[:n_trn]
# x_vld,y_vld=x_train[n_trn:n_trn+n_vld],y_train[n_trn:n_trn+n_vld]





# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(x_trn, y_trn, epochs=1
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
fl.client.start_numpy_client(server_address="server:22222", client=FlowerClient(), grpc_max_message_length = 1024*1024*1024)
