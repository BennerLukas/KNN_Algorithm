import pandas as pd
import numpy as np 
import sklearn
import math
import matplotlib.pyplot as plt
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


def valid(y_test, pred):
    'ermittle Richtigkeit für Confusion Matrix (TP,FN, TN, FP)'
    if pred == 1:
        if y_test == pred:
            return "TP"
        else:
            return "FP"
    else:
        if y_test==pred:
            return "TN"
        else:
            return  "FN"
        
def eval_results(results, anz):
    'berechne Kennzahlen zur Validierung des Systems'
    key = Counter(results).keys()
    values = Counter(results).values()
    eval_dict = dict(zip(key, values))

    print(eval_dict)

    try:
        tp= eval_dict["TP"]
    except:
        tp=0
    try:
        fn=eval_dict["FN"]
    except:
        fn=0
    try:
        tn = eval_dict["TN"]
    except:
        tn=0
    try:
        fp = eval_dict["FP"]
    except:
        fp=0


    sensitivity = tp / (tp+fn)
    precision = tp/(tp+fp)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    print(f"sensitivity:{sensitivity}, precision:{precision}, accuracy:{accuracy}")

#-------------------------------#
file_path= r".\data\mushrooms.csv"

raw_data = read(file_path)
data= label_encode((raw_data))
X_train, X_test, y_train, y_test = split_data(data)

y = y_train.tolist()
train = X_train.values.tolist()

results=[]
# len(X_test)
gesamt_anzahl= 10
preds = pd.Series()
for i in range(gesamt_anzahl):
    t0 = time.time()
    row = X_test.iloc[i].tolist()
    neighbours = get_neighbour(train, row, 10, y_train.iloc[i])
    pred = predict(neighbours)
    preds = preds.append(pd.Series([pred]))
    result = valid(y_test.iloc[i], pred)
    results.append(result)
    print(f"Zeit: {(time.time()-t0):.2}")


real = pd.DataFrame(data=y_test)
real = real[:gesamt_anzahl]
print(preds, real)
valid_data = real.insert(1,"pred",preds, allow_duplicates=True)
print(valid_data)
# class_report = metrics.classification_report()
eval_results(results, gesamt_anzahl)
