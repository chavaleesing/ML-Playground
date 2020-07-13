from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
clf = DecisionTreeClassifier().fit(X_train, y_train)

print(f"Accuracy of Decision Tree classifier on training set = {clf.score(X_train, y_train)}")
print(f"Accuracy of Decision Tree classifier on test set = {clf.score(X_test, y_test)}")
# result is overfitting ↑↑

# To avoid overfitting
clf = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)

print(f"Accuracy of Decision Tree classifier on training set = {clf.score(X_train, y_train)}")
print(f"Accuracy of Decision Tree classifier on test set = {clf.score(X_test, y_test)}")
