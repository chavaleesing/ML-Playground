import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from utils.dataset_generator import create_simple_regression_dataset


def demonstrate(params):
    X, y = create_simple_regression_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    reg = MLPRegressor(hidden_layer_sizes = [100,100],
                             activation = 'tanh',
                             alpha = 1,
                             solver = 'lbfgs').fit(X_train, y_train)

    accuracy = {
        "accuracy_train_set": reg.score(X_train, y_train),
        "accuracy_test_set": reg.score(X_test, y_test)
    }

    if params.get("plot"):
        fig, subaxes = plt.subplots(1, 1, figsize=(8,4))
        X_predict_input = np.linspace(-3, 3, 50).reshape(-1,1)
        y_predict_output = reg.predict(X_predict_input)
        subaxes.plot(X_predict_input, y_predict_output, '^', markersize = 10,
                    label='Predicted', alpha=0.8)
        subaxes.plot(X_train, y_train, 'o', label='True Value', alpha=0.8)
        subaxes.set_xlabel('Input feature')
        subaxes.set_ylabel('Target value')
        subaxes.set_title('NN regression')
        subaxes.legend()
        plt.tight_layout()
        fig.savefig('temp_plt.png', bbox_inches='tight')

    return accuracy
