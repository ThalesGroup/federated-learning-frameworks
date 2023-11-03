import collections
import os
import pandas as pd
import time
from matplotlib import pyplot as plt
from tensorflow import keras
import flwr as fl
import sys
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from flwr.common import Metrics

def load_data(path):
    data = pd.read_csv(path)
    columns = data.columns
    return data

def delete_features(dataFrame):
    df= dataFrame.drop(columns=['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp'])
    return df

def label_categorical_to_numeric(dataFrame):
    unique_labels = list(dataFrame.Label.astype('category').unique())
    unique_codes = list(dataFrame.Label.astype('category').cat.codes.unique())
    dataFrame['Label']= dataFrame['Label'].replace(unique_labels, unique_codes)
    dataFrame.rename(columns={"Label":"y"},inplace=True,errors="raise")

    # Replace Inf values with NaN
    dataFrame.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values
    dataFrame.dropna(inplace=True)

    return dataFrame


path_train = './IDDataset/Balanced_IEC104_Train_Test_CSV_Files/iec104_train_test_csvs/tests_cic_15/train_15_cicflow.csv'
path_test= './IDDataset/Balanced_IEC104_Train_Test_CSV_Files/iec104_train_test_csvs/tests_cic_15/test_15_cicflow.csv'

data_train=load_data(path_train)
data_test=load_data(path_test)

data_train=delete_features(data_train)
data_test=delete_features(data_test)

data_train=label_categorical_to_numeric(data_train)
data_test=label_categorical_to_numeric(data_test)

data_train.to_csv('./my_data_result/train_data.csv')
data_test.to_csv('./my_data_result/test_data.csv')