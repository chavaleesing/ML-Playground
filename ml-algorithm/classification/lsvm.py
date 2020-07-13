from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import os
import sys

sys.path.append(os.path.abspath(os.path.abspath(os.curdir)))
from dataset.dataset_generator import create_simple_classification_dataset


# Binary-class
X_C2, y_C2 = create_simple_classification_dataset()
X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2,
                                                   random_state = 0)
this_C = 0.1
clf = SVC(kernel = 'linear', C=this_C).fit(X_train, y_train)
print('Accuracy of SVC classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of SVC classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# Multi-class
