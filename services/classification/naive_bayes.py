import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from utils import plotter
from utils.dataset_generator import create_complex_classification_dataset, create_simple_classification_dataset


def demonstrate(params):
    X, y = create_simple_classification_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = GaussianNB().fit(X_train, y_train)
    accuracy = {
        "accuracy_train_set": clf.score(X_train, y_train),
        "accuracy_test_set": clf.score(X_test, y_test)
    }
    
    if params.get("plot"):
        fig = plt.figure()
        _, subaxes = plt.subplots(1, 1, figsize=(5, 5))
        subaxes = fig.add_subplot(1, 1, 1)
        plotter.plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, 'Naive Bayes classification for binary dataset', subaxes)
        subaxes.set_xlabel('height')
        subaxes.set_ylabel('width')
        cv = FigureCanvasAgg(fig)
        cv.print_png('temp_plt.png')
    return accuracy
