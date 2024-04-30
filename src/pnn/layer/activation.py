import numpy as np
from typing import Callable

from pnn.tensor import Tensor
from .layer import Layer


class ActivationLayer(Layer):
    def __init__(self, activation: Callable) -> None:
        self.activation = activation

    def forward(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):
        for i in range(0, len(in_tensors)):
            self.activation(in_tensor=in_tensors[i], out_tensor=out_tensors[i])

    def backward(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        pass

def sigmoid(in_tensor: Tensor, out_tensor: Tensor):
    sigmoid_lambda = lambda x: 1/(1 + np.exp(-x))
    sigmoid_func = np.vectorize(sigmoid_lambda)
    out_tensor = Tensor(sigmoid_func(in_tensor.elements), None)

def relu(in_tensor: Tensor, out_tensor: Tensor):
    pass

def tanh(in_tensor: Tensor, out_tensor: Tensor):
    pass

def soft_max(in_tensor: Tensor, out_tensor: Tensor):
    summe = np.sum(np.exp(in_tensor.elements))
    softmax_lambda = lambda x_i: np.exp(x_i)/summe
    softmax_func = np.vectorize(softmax_lambda)
    # softmax_func /= summe
    out_tensor.elements = softmax_func(in_tensor.elements)
