import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        sum=0;
        max = np.max(z)
        for i in z:
            sum =sum + np.exp(i-max)
        return np.round(np.exp(z-max)/sum, 4)
