from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

sys.path.append(os.path.abspath(os.path.abspath(os.curdir)))
from dataset.dataset_generator import create_simple_regression_dataset


X, y = create_simple_regression_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# linreg = Ridge(alpha=20).fit(X_train, y_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)

print(f"y-interception = {linridge.intercept_}, slope = {linridge.coef_}")
print(f'R-squared score (training): {linridge.score(X_train_scaled, y_train)}')
print(f'R-squared score (test): {linridge.score(X_test_scaled, y_test)}')
