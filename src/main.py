#Import der Bibliotheken
import pandas as pd
import numpy as np 

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#Definitionen der Funktionen
def read(file_path):
    'Lese die Pilz-Daten ein'
    data = pd.read_csv(file_path, sep=",")
    data=data[["class","cap-shape","odor"]]
    return data

def split_data(data):
    'Input LabelEcode Daten: erstelle Test und Trainingsdatensätze'
    y = data["class"]
    X = data[["cap-shape","odor"]]
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, test_size=0.3, random_state=5)
    return X_train, X_test, y_train, y_test

def label_encode(data):
    'Label-Encode Daten -> aus String wird einzigartige Zahl'
    encoder = LabelEncoder()
    for  col in data.columns:
        data[col]=encoder.fit_transform(data[col])
    return data

def rescaling(data, param): 
    'skaliert jede Zahl auf einen Wert zwischen 0 und 1. Notwendig für Abstandsbestimmungen'
    scaler = sklearn.preprocessing.MinMaxScaler()
    data[param] = scaler.fit_transform(data[param])
    return data

def knn_model(k=5):
    'erstelle das KNN Model mit seinen Parametern'
    knn = KNeighborsClassifier(n_neighbors=k,metric="euclidean",algorithm="brute")
    # print(knn)
    return knn

def train_test(knn, X_train, X_test, y_train, y_test):
    '"trainiere" das KNN_Model mit Trainingsdaten und Teste es auf neuen Daten'
    knn.fit(X_train, y_train)
    y_prediction = knn.predict(X_test)
    accurarcy = metrics.accuracy_score(y_test,  y_prediction)
    class_report = metrics.classification_report(y_test, y_prediction)
    print("_"*60)
    print("das verwendete Modell hat folgende Eigenschaften:")
    print(knn)
    print("_"*60)
    print("Evaluierung des Models:")
    print(class_report)
    print(f"Die Genauigkeit beträgt: {accurarcy}")
    print("_"*60)
    return accurarcy



#------------Main---------------#
file_path= "data/mushrooms.csv" # Pfad muss gegebenenfalls angepasst werden

#Datenvorbereitung
raw_data = read(file_path)
data = label_encode(raw_data)
data = rescaling(data, ["cap-shape","odor"])
(X_train, X_test, y_train, y_test) = split_data(data)

#Modellerstellung
knn = knn_model()

#Modeltraining und Validierung mit für Modell unbekannten Testdaten
train_test(knn,X_train, X_test,y_train, y_test)



