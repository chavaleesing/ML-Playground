import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from utils import plotter


def demonstrate(params):
    default_path = os.path.abspath(os.getcwd()) 
    path = f"{default_path}/dataset/fruit.csv"
    data_frames = pd.read_csv(path, delimiter=",,", engine='python')
    X = data_frames[['height', 'width', 'mass' ,'color_score']]
    y = data_frames['fruit_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = KNeighborsClassifier(n_neighbors = 5)
    clf.fit(X_train, y_train)

    accuracy = {
        "accuracy_train_set": clf.score(X_train, y_train),
        "accuracy_test_set": clf.score(X_test, y_test)
    }

    if params.get("plot"):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)
        ax.set_xlabel('width')
        ax.set_ylabel('height')
        ax.set_zlabel('color_score')
        cv = FigureCanvasAgg(fig)
        cv.print_png('temp_plt.png')

    return accuracy
