from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.abspath(os.curdir)))
from dataset.dataset_generator import create_simple_regression_dataset


X, y = create_simple_regression_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
print(X, y)
linreg = LinearRegression().fit(X_train, y_train)

print(f"y-interception = {linreg.intercept_}, slope = {linreg.coef_}")
print(f'R-squared score (training): {linreg.score(X_train, y_train)}')
print(f'R-squared score (test): {linreg.score(X_test, y_test)}')

## Plot graph
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.scatter(X, y, marker= '.') # plot dataset
plt.plot(X, linreg.coef_ * X + linreg.intercept_, color='green') # plot predicted graph
plt.title('Least-squares linear regression')
plt.xlabel('Feature value (x)')
plt.ylabel('Target value (y)')
plt.show()
