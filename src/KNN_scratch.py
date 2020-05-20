import pandas as pd
import numpy as np 
import sklearn
import  math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from collections import  Counter

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



# calculate_euclidean_distance
def calc_distance(row1, row2):
    distance=0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i] )**2
    return math.sqrt(distance)

def get_neighbour(train, row_test, k,y):
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
    y= [i[-1] for i in neighbours]
    pred = max(set(y), key=y.count) # gibt häufigste Klasse
    return pred


def valid(y_test, pred):
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
    key = Counter(results).keys()
    values = Counter(results).values()

    

    print(key, values)

#-------------------------------#
file_path= r".\data\mushrooms.csv"

raw_data = read(file_path)
X_train, X_test, X_valid, y_train, y_test, y_valid = prepare_data(raw_data)

y = y_train.tolist()
train = X_train.values.tolist()
row = X_test.iloc[0].tolist()

results=[]
gesamt_anzahl=5
for i in range(gesamt_anzahl):
    row = X_test.iloc[i].tolist()
    neighbour = get_neighbour(train, row, 5, y_train.iloc[i])
    pred = predict(neighbour)
    results.append(valid(y_test.iloc[i], pred))
    print(pred, y_test.iloc[i])

eval_results(results, gesamt_anzahl)
