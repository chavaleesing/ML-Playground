from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from utils.dataset_generator import create_classification_iris_dataset
from utils import plotter


def demonstrate(params):
    dataset = params.get("dataset", "iris") # temp force to use iris dataset
    if dataset == 'iris':
        ds = create_classification_iris_dataset()
    X_train, X_test, y_train, y_test = train_test_split(ds.data, ds.target)
    clf = DecisionTreeClassifier(max_depth=int(params.get("max_depth", 3))).fit(X_train, y_train)

    accuracy = {
        "accuracy_train_set": clf.score(X_train, y_train),
        "accuracy_test_set": clf.score(X_test, y_test)
    }

    if params.get("plot"):
        plotter.plot_decision_tree(clf, ds.feature_names, ds.target_names)

    return accuracy
