import numpy as np
from ..tensor import Tensor
from .layer import Layer
from typing import Callable

class LossLayer(Layer):
    def __init__(self, loss: Callable) -> None:
        self.loss = loss

    def forward(self, predictions: list[Tensor], targets: list[Tensor]):
        # return np.sum([self.loss(target=targets[i], prediction=predictions[i]) 
        #                for i in range(len(targets))])
        lossl = []
        for i in range(len(targets)):
            lossl.append(self.loss(target=targets[i], prediction=predictions[i]))
        return np.sum(lossl)
        
    def backward(self, predictions: list[Tensor], targets: list[Tensor]):
        for i in range(len(targets)):
            predictions[i].deltas = self.loss(target=targets[i], prediction=predictions[i], forward = False) 
                

def mean_squared_error(target: Tensor, prediction: Tensor) -> float:  
    #if forward:
    return np.sum(np.divide(np.power(target.elements - prediction.elements, 2), len(target.elements)))
    # else:  
    #     return prediction.elements - target.elements


def cross_entropy(target: Tensor, prediction: Tensor, forward: bool = True):
    return - np.sum(target.elements * np.log(prediction.elements))