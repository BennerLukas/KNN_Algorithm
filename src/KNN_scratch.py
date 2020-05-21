import pandas as pd
import numpy as np 
import sklearn
import  math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import  Counter
import time

def read(file_path):
    'Lese die Pilz-Daten ein'
    # file_path= r".\data\mushrooms.csv"
    data = pd.read_csv(file_path, sep=",")
    return data

def prepare_data(data):
    data = label_encode(data)
    y = data["class"]
    X = data.drop(["class"],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, test_size=0.2, random_state=5)
    
    return X_train, X_test, y_train, y_test

def label_encode(data):

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
    t1 = time.time()
    distances = []
    for row in train:
        dist=calc_distance(row_test, row)
        distances.append((row,dist,y))
    distances.sort(key=lambda t: t[1]) #sortiert nach der zweiten Stelle im Tupel
    neighbours = []
    for i in range(k):
        neighbours.append(distances[i])
    print(f"get_Neighbour: {(time.time()-t1):.6f}")
    return neighbours

def predict(neighbours):
    'ermittelt aus y der Nachbarn das y des zu bestimmtenden Pilzes'
    y= [i[-1] for i in neighbours]
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

    tp= eval_dict["TP"]
    fn=eval_dict["FN"]
    tn = eval_dict["TN"]
    fp = eval_dict["FP"]

    sensitivity = tp / (tp+fn)
    precision = tp/(tp+fp)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    print(sensitivity, precision, accuracy)

#-------------------------------#
file_path= r".\data\mushrooms.csv"

raw_data = read(file_path)
X_train, X_test, y_train, y_test = prepare_data(raw_data)

y = y_train.tolist()
train = X_train.values.tolist()

results=[]
gesamt_anzahl=100
for i in range(gesamt_anzahl):

    t0 = time.time()

    row = X_test.iloc[i].tolist()
    neighbour = get_neighbour(train, row, 5, y_train.iloc[i])
    pred = predict(neighbour)
    result = valid(y_test.iloc[i], pred)
    # print(result)
    results.append(result)
    
    print(f"Zeit: {(time.time()-t0):.5f}")

eval_results(results, gesamt_anzahl)
