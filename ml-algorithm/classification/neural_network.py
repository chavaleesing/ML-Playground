from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split


X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                       centers = 8, cluster_std = 1.3,
                       random_state = 4)
X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

for units in [1, 10, 100]:
    nnclf = MLPClassifier(hidden_layer_sizes = [units], solver='lbfgs',
                         random_state = 0).fit(X_train, y_train)
