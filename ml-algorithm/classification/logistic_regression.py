#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import pandas as pd

sys.path.append('/Users/chavalee/ings/ML-Playground')
from utils.dataset_generator import create_simple_classification_dataset
# from utils.plotter import plot_class_regions_for_classifier_subplot


fig, subaxes = plt.subplots(1, 1, figsize=(5, 5))
X_C2, y_C2 = create_simple_classification_dataset()
X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state=0)
print(X_train[0:5])
clf = LogisticRegression().fit(X_train, y_train)
cs = clf.score(X_test, y_test)
print(f'Accuracy of Logistic regression classifier on training set: {clf.score(X_train, y_train)}')
print(f'Accuracy of Logistic regression classifier on test set: {clf.score(X_test, y_test)}')

def plot_class_regions_for_classifier_subplot(clf, X, y, X_test, y_test, title, subplot):
    numClasses = np.amax(y) + 1

    color_list_light = ['#FFFFAA', '#EFEFEF', '#AAFFAA', '#AAAAFF']
    color_list_bold = ['#EEEE00', '#000000', '#00CC00', '#0000CC']
    cmap_light = ListedColormap(color_list_light[0:numClasses])
    cmap_bold  = ListedColormap(color_list_bold[0:numClasses])

    h = 0.03
    k = 0.5
    x_plot_adjust = 0.1
    y_plot_adjust = 0.1
    plot_symbol_size = 50

    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    print(x_min,x_max,y_min,y_max)
    x2, y2 = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    xy = np.c_[x2.ravel(), y2.ravel()]
    print(xy)
    P = clf.predict(xy)
    print(P, len(P),x2.shape)
    P = P.reshape(x2.shape)
    print(len(P[0]))

    subplot.contourf(x2, y2, P, cmap=cmap_light, alpha = 0.8)

    subplot.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, s=plot_symbol_size, edgecolor = 'black')
    subplot.set_xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
    subplot.set_ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)
    subplot.set_title(title)


plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, 'Logistic regression for binary dataset', subaxes)
subaxes.set_xlabel('height')
subaxes.set_ylabel('width')



# %%
