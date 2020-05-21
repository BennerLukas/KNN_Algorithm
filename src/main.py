import pandas as pd
import numpy as np #
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def read(file_path):
    'Lese die Pilz-Daten ein'
    # file_path= r".\data\mushrooms.csv"
    data = pd.read_csv(file_path, sep=",")
    return data



def prepare_data(data):
    'Label encode Daten und erstelle Test und Trainingsdatensätze'
    data = label_encode(data)
    y = data["class"]
    X = data.drop(["class"],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, test_size=0.3, random_state=5)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, train_size=0.7, random_state=5)

    return X_train, X_test, X_valid, y_train, y_test, y_valid

def label_encode(data):
    'Label-Encode Daten -> aus String wird einzigartige Zahl'
    encoder = LabelEncoder()
    for  col in data.columns:
        data[col]=encoder.fit_transform(data[col])
    return data

def visualize_data(data):
    pass

def create_scatter_plot():
    pass

def knn_model(k=5):
    'erstelle das KNN Model mit seinen Parametern'
    knn = KNeighborsClassifier(n_neighbors=k,metric="euclidean")
    return knn
def training(knn, X_train, X_test, y_train, y_test):
    'trainiere das KNN_Model mit Trainingsdaten und Teste es auf neuen Daten'
    knn.fit(X_train, y_train)
    y_prediction = knn.predict(X_test)
    # accurarcy = metrics.accuracy_score(y_test,  y_prediction)
    conf_matrix = metrics.confusion_matrix(y_test, y_prediction)
    class_report = metrics.classification_report(y_test, y_prediction)
    print(conf_matrix)
    print(class_report)


def valid():
    'validiere das Model abschließend auf vollkommen unbekannten Daten'
    pass


#------------Main---------------#
file_path= r".\data\mushrooms.csv"

raw_data = read(file_path)
X_train, X_test, X_valid, y_train, y_test, y_valid = prepare_data(raw_data)
knn = knn_model()
training(knn,X_train, X_test,y_train, y_test)