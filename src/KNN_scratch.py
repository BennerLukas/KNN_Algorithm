#Import der Bibliotheken

import pandas as pd
import numpy as np 

#(nur verwendet zum Datenvorbereiten und Validieren)
# Algorithmus nur mithilfe von numpy und pandas geschrieben
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

import time

def read(file_path):
    'Lese die Pilz-Daten ein'
    data = pd.read_csv(file_path, sep=",")
    return data


def split_data(data):
    'Input LabelEcode Daten: erstelle Test und Trainingsdatensätze'
    y = data["class"]
    X = data[["cap-shape","odor"]]
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, test_size=0.3, random_state=123)
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

def calc_distance(x1, x2):
    'berechnet euklidischen Abstand mithilfe des Betrags der Subtraktion zweier Vektoren'
    distance = np.linalg.norm(x1-x2)
    return distance

def get_neighbour(points, x1, k,y):
    'ermittelt die k nächsten Nachbarn und gibt sie in Liste zurück'    
    distances = []
    for i in range(len(points)):
        dist=calc_distance(x1, points[i])
        distances.append((dist,y.iloc[i]))
    distances.sort(key=lambda t: t[0]) #sortiert nach der Ersten Stelle im Tupel
    return distances[:k]


def predict(neighbours):
    'ermittelt aus y der Nachbarn das y des zu bestimmtenden Pilzes'
    y= [i[-1] for i in neighbours] # gibt die Klasse aller Nachbarn in Liste aus
    pred = max(set(y), key=y.count) # gibt häufigste Klasse
    return int(pred)

def eval_results(preds,real_values):
    'gibt verschiedene Validationswerte zurück'
    accurarcy = metrics.accuracy_score(real_values,preds)
    class_report = metrics.classification_report(real_values,preds)
    print(class_report)
    print(f"accuracy:{accurarcy}")

#-------------------------------#

file_path= "data/mushrooms.csv" # Pfad muss gegebenenfalls angepasst werden

#Datenvorbereitung
raw_data = read(file_path)
data = label_encode((raw_data))
data = rescaling(data,["cap-shape","odor"])
X_train, X_test, y_train, y_test = split_data(data)

#Parameter festlegen
bereich = 100 #len(X_train)
k=5



real_values = y_test.iloc[:bereich].tolist() # echte Zielvariablen
preds = [] # Vorhersage der Zielvariable

#Nachbarberechnung und Vorhersage
for i in range(bereich):
    neighbours = get_neighbour(X_train.to_numpy(), X_test.iloc[i].to_numpy(), k, y_train )
    pred = predict(neighbours)
    preds.append(pred)

print(preds)
print(real_values)

#Evaluierung
eval_results(preds, real_values)

