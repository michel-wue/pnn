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
        for i in range(0, len(in_tensors)):
            self.activation(in_tensor=in_tensors[i], out_tensor=out_tensors[i], forward=False)


def _raw_sigmoid(in_tensor: Tensor):
    return np.divide(1, (np.add(1, np.exp(-in_tensor.elements))))


def sigmoid(in_tensor: Tensor, out_tensor: Tensor, forward: bool = True):
    if forward:
        out_tensor.elements = _raw_sigmoid(in_tensor)
    else:
        in_tensor.deltas = np.multiply(np.multiply(_raw_sigmoid(in_tensor), (1 - _raw_sigmoid(in_tensor))),
                                       out_tensor.deltas)


def _raw_soft_max(in_tensor: Tensor):
    summe = np.sum(np.exp(in_tensor.elements))
    return np.divide(np.exp(in_tensor.elements), summe)


def soft_max(in_tensor: Tensor, out_tensor: Tensor, forward: bool = True):
    if forward:
        out_tensor.elements = _raw_soft_max(in_tensor)
    else:
        length = len(out_tensor.elements)
        matrix1 = np.zeros(shape=(length, length))
        matrix2 = np.zeros(shape=(length, length))
        for i in range(length):
            for j in range(length):
                if i == j:
                    matrix1[i][j] = out_tensor.elements[i]

                matrix2[i][j] = out_tensor.elements[i] * out_tensor.elements[j]

        in_tensor.deltas = np.dot(out_tensor.deltas, matrix1 - matrix2)

        # in_tensor.deltas = np.multiply(_raw_soft_max(in_tensor), out_tensor.deltas)


def relu(in_tensor: Tensor, out_tensor: Tensor, forward: bool = True):
    if forward:
        out_tensor.elements = _raw_relu(in_tensor)
    else:
        in_tensor.deltas = np.where(out_tensor.deltas > 0, 1, 0)


def _raw_relu(in_tensor: Tensor):
    return np.maximum(0, in_tensor.elements)


def tanh(in_tensor: Tensor, out_tensor: Tensor):
    pass
