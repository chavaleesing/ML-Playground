from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils import plotter
from utils.dataset_generator import create_classification_iris_dataset


def demonstrate(params):
    dataset = params.get("dataset", "iris")
    max_depth = int(params.get("max_depth", 3))
    if dataset == 'iris' or True:  # force to use iris dataset
        ds = create_classification_iris_dataset()
        X, y = ds.data, ds.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)

    accuracy = {
        "accuracy_train_set": clf.score(X_train, y_train),
        "accuracy_test_set": clf.score(X_test, y_test),
        "feature_importances": clf.feature_importances_
    }

    if params.get("plot"):
        plotter.plot_decision_tree(clf, ds.feature_names, ds.target_names)

    return accuracy
