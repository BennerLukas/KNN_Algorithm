import pandas as pd
import numpy as np 
import sklearn
import matplotlib.pyplot as plt
import seaborn as sbn 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from matplotlib.colors import ListedColormap

import seaborn as sns

def read(file_path):
    'Lese die Pilz-Daten ein'
    # file_path= r".\data\mushrooms.csv"
    data = pd.read_csv(file_path, sep=",")
    data=data[["class","cap-shape","odor"]]
    return data



def split_data(data):
    'Input LabelEcode Daten: erstelle Test und Trainingsdatensätze'
    y = data["class"]
    X = data[["cap-shape","odor"]]
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, test_size=0.3, random_state=5)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, train_size=0.9, random_state=5)

    return X_train, X_test, X_valid, y_train, y_test, y_valid

def label_encode(data):
    'Label-Encode Daten -> aus String wird einzigartige Zahl'
    encoder = LabelEncoder()
    for  col in data.columns:
        data[col]=encoder.fit_transform(data[col])
    return data

def visualize_data(data):
    plot_data=data[["cap-shape","odor"]]
    noise = np.random.normal(0,0.1,plot_data.shape)
    plot_data = plot_data +noise    #für bessere Ansicht
    create_scatter_plot(plot_data,data["class"])
    

def create_scatter_plot(X,y, name="Scatter-Plot", xlabel=None, ylabel=None):
    plt.title("Geruch - Form des Pilzhuts Diagramm ")
    colour=ListedColormap(["g","r"])
    scatter =plt.scatter(X["odor"],X["cap-shape"],c=y, cmap=colour, alpha=0.2)
    plt.xlabel("Geruch")
    plt.ylabel("Form")
    plt.legend(handles=scatter.legend_elements()[0], labels=["essbar","giftig"])
    plt.show()
    plt.close()

def knn_model(k=5):
    'erstelle das KNN Model mit seinen Parametern'
    knn = KNeighborsClassifier(n_neighbors=k,metric="euclidean",algorithm="brute")
    print(knn)
    return knn
def train(knn, X_train, X_test, y_train, y_test):
    'trainiere das KNN_Model mit Trainingsdaten und Teste es auf neuen Daten'
    knn.fit(X_train, y_train)
    y_prediction = knn.predict(X_test)
    accurarcy = metrics.accuracy_score(y_test,  y_prediction)
    conf_matrix = metrics.confusion_matrix(y_test, y_prediction)
    class_report = metrics.classification_report(y_test, y_prediction)
    #precision = metrics.precision_score(y_test,y_prediction)
    print(conf_matrix)
    print(class_report)
    print(f"accuracy:{accurarcy}")
    return accurarcy

def plot_changes_with_k(list_of_k,X_train, X_test, y_train, y_test):
    "Schaut sich an wie die Genauigkeit sich bei Veränderung von k verhält"
    scores = []
    for x in list_of_k:
        knn = knn_model(x)
        score = train(knn,X_train, X_test,y_train, y_test)
        print(X_train.shape)
        scores.append(score)
    plt.plot(list_of_k,scores,"-o")
    plt.title("Auswirkung der Veränderung von k auf die Genauigkeit")
    plt.ylabel("Genauigkeit")
    plt.xlabel("K")
    #plt.show()
    plt.savefig("Veränderung_K.png")


def valid(X_valid, y_valid, knn):
    'validiere das Model abschließend auf vollkommen unbekannten Daten'
    y_prediction = knn.predict(X_valid)
    accurarcy = metrics.accuracy_score(y_valid,  y_prediction)
    conf_matrix = metrics.confusion_matrix(y_valid, y_prediction)
    class_report = metrics.classification_report(y_valid, y_prediction)
    print(conf_matrix)
    print(class_report)
    print(f"accuracy:{accurarcy}")   


#------------Main---------------#
file_path= r".\data\mushrooms.csv"
file_path= "data/mushrooms.csv"

raw_data = read(file_path)
data = label_encode(raw_data)
visualize_data(data)

(X_train, X_test, X_valid, y_train, y_test, y_valid) = split_data(data)

knn = knn_model()
train(knn,X_train, X_test,y_train, y_test)

#list_of_k = [1,3,5,10,25,50,75,100]
#plot_changes_with_k(list_of_k,X_train, X_test, y_train, y_test)

print("Validation:")
valid(X_valid, y_valid, knn)