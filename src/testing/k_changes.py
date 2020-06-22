import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
import seaborn as sbn




dataset = datasets.load_iris()
X, y = dataset.data, dataset.target

# we only take two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = X[:, [0, 2]]

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, stratify=y, test_size=0.7, random_state=42)

list_of_neigbors = [1,2,5,6,20,20,45]

def plot_differnet_scores(list_of_neigbors,X_train,X_test,y_train,y_test):
    scores = []
    for x in list_of_neigbors:
        knn = KNeighborsClassifier(n_neighbors=x)
        model = knn.fit(X_train,y_train)
        a = model.score(X_test,y_test)
        scores.append(a)

    sbn.scatterplot(x=scores, y=list_of_neigbors)
    plt.show()

plot_differnet_scores(list_of_neigbors,X_train,X_test,y_train,y_test)


