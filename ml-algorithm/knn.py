import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

default_path = os.path.abspath(os.getcwd()) 

path = f"{default_path}/dataset/fruit.csv"
data_frames = pd.read_csv(path, delimiter=",,")
# print(data_frames.head())

X = data_frames[['height', 'width', 'mass', 'color_score']]
y = data_frames['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train) # training data
knn.score(X_test, y_test) # evaluating knn model

fruit_prediction = knn.predict([[100, 6.3, 6.8, 0.55]])
print(fruit_prediction)
