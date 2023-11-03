import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Load and compile Keras model
def model_def(a,b):
    model=tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(a,)),
    tf.keras.layers.Dense(units=50, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=b, activation='softmax')])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model