from .layer import Layer
from ..tensor import Tensor
from ..shape import Shape
import numpy as np

class FullyConnected(Layer):
    def __init__(
            self, 
            out_shape: Shape,
            in_shape: Shape = None,
            ) -> None:
        self.out_shape = out_shape
        self.in_shape = in_shape
        # set input shape in Network
        self.bias: Tensor = None
        self.weights: Tensor = None
        
    # overwrite 
    def forward(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):
        for i in range(0, len(in_tensors)):
            out_matrix = np.dot(in_tensors[i].elements, self.weights.elements) + self.bias.elements
            out_tensors[i] = Tensor(elements=out_matrix, deltas=None)
    
    # overwrite
    def backward(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        for i in range(0, len(in_tensors)):
            in_deltas = np.dot(out_tensors[i].deltas, self.weights.elements.T)
            in_tensors[i].deltas = in_deltas

    # overwrite
    def calculate_delta_weights(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        x_matrix = np.array([in_tensor.elements for in_tensor in in_tensors])
        dY_matrix = np.array([out_tensor.deltas for out_tensor in out_tensors])
        # return x_matrix, dY_matrix
        self.weights.deltas = np.dot(x_matrix.T, dY_matrix)
        self.bias.deltas = dY_matrix
        # delta_weights = np.zeros_like(self.weights.elements)
        # delta_bias = np.zeros_like(self.bias.elements)
        # for i in range(len(in_tensors)):
        #     delta_weights += np.dot(in_tensors[i].elements.T, out_tensors[i].deltas)
        #     delta_bias += out_tensors[i].deltas
        # self.weights.deltas = delta_weights
        # self.bias.deltas = delta_bias

    def save_params():
        pass

    def load_params():
        pass
