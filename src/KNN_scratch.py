import pandas as pd
import numpy as np 
import sklearn
import math
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import  Counter
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

    sbn.scatterplot(x=data["cap-shape"], y=data.odor, hue=data["class"])
    plt.show()
    
    return X_train, X_test, y_train, y_test

def label_encode(data):
    'Label-Encode Daten -> aus String wird einzigartige Zahl'
    encoder = LabelEncoder()
    for  col in data.columns:
        data[col]=encoder.fit_transform(data[col])
    return data

def calc_distance(row1, row2):
    'berechnet euklidischen Abstand: sqrt((x1-x2)**2)'
    distance=0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i] )**2
    return math.sqrt(distance)

def get_neighbour(train, row_test, k,y):
    'ermittelt die k nächsten Nachbarn und gibt sie in Liste zurück'    
    distances = []
    for row in train:
        dist=calc_distance(row_test, row)
        distances.append((row,dist,y))
    distances.sort(key=lambda t: t[1]) #sortiert nach der zweiten Stelle im Tupel
    neighbours = []
    for i in range(k):
        neighbours.append(distances[i])
    return neighbours

def predict(neighbours):
    'ermittelt aus y der Nachbarn das y des zu bestimmtenden Pilzes'
    neighbours
    y= [i[-1] for i in neighbours] # gibt die Klasse aller Nachbarn in Liste aus
    pred = max(set(y), key=y.count) # gibt häufigste Klasse
    return pred

def eval_results(preds,real_values):
    'gibt verschiedene Validationswerte zurück'
    accurarcy = metrics.accuracy_score(real_values,preds)
    conf_matrix = metrics.confusion_matrix(real_values,preds)
    class_report = metrics.classification_report(real_values,preds)
    #precision = metrics.precision_score(real_values,preds)
    print(conf_matrix)
    print(class_report)
    print(f"accuracy:{accurarcy}")

#-------------------------------#
file_path= r".\data\mushrooms.csv"
file_path= "data/mushrooms.csv"

raw_data = read(file_path)
data= label_encode((raw_data))
X_train, X_test, y_train, y_test = split_data(data)

y = y_train.tolist()
train = X_train.values.tolist()


gesamt_anzahl= 1000
preds = pd.Series()
real_values = pd.Series()
for i in range(gesamt_anzahl):
    t0 = time.time()
    row = X_test.iloc[i].tolist()
    neighbours = get_neighbour(train, row, 10, y_train.iloc[i])
    pred = predict(neighbours)
    preds = preds.append(pd.Series([pred]))
    real_value = y_test.iloc[i]
    real_values = real_values.append(pd.Series([real_value]))
    #print(f"Zeit: {(time.time()-t0):.2}")

eval_results(preds,real_values)
# real = pd.DataFrame(data=y_test)
# real = real[:gesamt_anzahl]
# print(preds, real)
# valid_data = real.insert(1,"pred",preds, allow_duplicates=True)
# print(valid_data)
# class_report = metrics.classification_report()