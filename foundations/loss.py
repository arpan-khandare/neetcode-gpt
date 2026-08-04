import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        sum_1=0
        sum_0=0
        for i in range(y_true.size):
            if y_true[i]==1:
                sum_1 += np.log(y_pred[i])
            else:
                sum_0 += np.log(1-y_pred[i])
        return round(-(1/y_true.size)*(sum_1+sum_0), 4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        return round(-(1/y_true.shape[0])*np.sum(y_true*np.log(y_pred)), 4)
        
