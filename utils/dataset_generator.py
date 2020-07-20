from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


# synthetic dataset for simple regression
def create_simple_regression_dataset(plot=False):
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples = 100, n_features=1,
                                n_informative=1, bias = 150.0,
                                noise = 30, random_state=0)
    if plot:
        plt.figure()
        plt.title('Sample regression problem with one input variable')                            
        plt.scatter(X, y, marker= 'o', s=50)
        plt.show()
    return X, y


# synthetic dataset for more complex regression
def create_complex_regression_dataset(plot=False):
    from sklearn.datasets import make_friedman1
    X, y = make_friedman1(n_samples = 100,
                            n_features = 7, random_state=0)
    if plot:
        plt.figure()
        plt.title('Complex regression problem with one input variable')
        plt.scatter(X[:, 2], y, marker= 'o', s=50)
        plt.show()
    return X, y


# synthetic dataset for classification (binary) 
def create_simple_classification_dataset(plot=False):
    X, y = make_classification(n_samples = 100, n_features=2,
                                    n_redundant=0, n_informative=2,
                                    n_clusters_per_class=1, flip_y = 0.1,
                                    class_sep = 0.5, random_state=0)
    if plot:
        cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])
        plt.figure()
        plt.title('Sample binary classification problem with two informative features')
        plt.scatter(X[:, 0], X[:, 1], c=y,
                marker= 'o', s=50, cmap=cmap_bold)
        plt.show()
    return X,y


# more difficult synthetic dataset for classification (binary) 
# with classes that are not linearly separable
def create_complex_classification_dataset(plot=False):
    X, y = make_blobs(n_samples = 100, n_features = 2, centers = 8,
                        cluster_std = 1.3, random_state = 4)
    y = y % 2
    if plot:
        cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])
        plt.figure()
        plt.title('Sample binary classification problem with non-linearly separable classes')
        plt.scatter(X[:,0], X[:,1], c=y,
                marker= 'o', s=50, cmap=cmap_bold)
        plt.show()
    return X,y


# Breast cancer dataset for classification
def create_classification_cancer_dataset():
    cancer = load_breast_cancer()
    (X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
    return (X_cancer, y_cancer)

def create_classification_iris_dataset():
    from sklearn.datasets import load_iris
    iris = load_iris()
    return iris.data, iris.target