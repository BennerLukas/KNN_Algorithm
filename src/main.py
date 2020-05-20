import pandas as pd
import numpy as np #
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def read(file_path):
    'Lese die Pilz-Daten ein'
    # file_path= r".\data\mushrooms.csv"
    data = pd.read_csv(file_path, sep=",")
    return data



def prepare_data(data):
    data = label_encode(data)
    y = data["class"]
    X = data.drop(["class"],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, test_size=0.3, random_state=5)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, train_size=0.7, random_state=5)

    return X_train, X_test, X_valid, y_train, y_test, y_valid

def label_encode(data):

    encoder = LabelEncoder()
    for  col in data.columns:
        data[col]=encoder.fit_transform(data[col])
    return data

def visualize_data():
    pass

def knn_model():
    pass

def train():
    pass

def test():
    pass


def valid():
    pass


#-------------------------------#
file_path= r".\data\mushrooms.csv"

raw_data = read(file_path)
X_train, X_test, X_valid, y_train, y_test, y_valid = prepare_data(raw_data)
