import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

default_path = os.path.abspath(os.getcwd()) 

path = f"{default_path}/dataset/fruit.csv"
data_frames = pd.read_csv(path, delimiter=",,", engine='python')
# print(data_frames.head())

X = data_frames[['height', 'width', 'mass', 'color_score']]
y = data_frames['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train) # training data
knn.score(X_test, y_test) # evaluating knn model

# predict in order of ['height', 'width', 'mass', 'color_score']
fruit_prediction = knn.predict([[100, 6.3, 6.8, 0.55]])
print(f"Fruit Prediction is: {fruit_prediction}")


# plotting a 3D scatter plot
print("--------------------------------------------------------------- Plot Graph 3D ---")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
colors = y_train.map({'lemon':'r', 'apple':'g', 'orange':'b', 'mandarin':'#E240D2'})
ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = colors, marker = 'o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
# plt.show()
print("--------------------------------------------------------------- Plot Graph 3D ---")
