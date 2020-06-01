import pandas as pd
import numpy as np 
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import seaborn as sns

def read(file_path):
    'Lese die Pilz-Daten ein'
    # file_path= r".\data\mushrooms.csv"
    data = pd.read_csv(file_path, sep=",")
    return data



def split_data(data):
    'Input LabelEcode Daten: erstelle Test und Trainingsdatensätze'
    data = data[["class", "cap-shape","odor"]]
    y = data["class"]
    X = data.drop(["class"],axis=1)
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
    print(data.head())
    plot_data= data[["odor", "cap-color"]]
    
    noise = np.random.normal(0,0.1,plot_data.shape)
    plot_data = plot_data +noise
    create_scatter_plot(plot_data,data["class"])
    
    # col=data.columns.values[:11]
    # # data= data[col]
    # print(data)
    # sns.pairplot(data, hue="class")
    # plt.savefig("Scatter_Matrix")
    # plt.show()

def create_scatter_plot(X,y, name="Scatter-Plot", xlabel=None, ylabel=None):
    plt.scatter(X["odor"],X["cap-color"],c=y, alpha=0.2)
    
    plt.show()

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
    print(conf_matrix)
    print(class_report)
    print(f"accuracy:{accurarcy}")


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

raw_data = read(file_path)
data = label_encode(raw_data)
# visualize_data(data)


(X_train, X_test, X_valid, y_train, y_test, y_valid) = split_data(data)
knn = knn_model()
train(knn,X_train, X_test,y_train, y_test)



# print("Validation:")
# valid(X_valid, y_valid, knn)