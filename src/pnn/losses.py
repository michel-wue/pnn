import numpy as np

def mean_squared_error(actual: np.array, prediction: np.array) -> float:  
    mse = np.array([1/2*(actual[i]-prediction[i])**2 for i in range(len(actual))])
    return np.sum(mse)

def cross_entropy():
    pass