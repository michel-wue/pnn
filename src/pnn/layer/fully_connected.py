from .layer import Layer
from ..tensor import Tensor
from ..shape import Shape
import numpy as np

class FullyConnected(Layer):
    def __init__(
            self, 
            out_shape: Shape,
            initialization_technique: str = None, 
            in_shape: Shape = None,
            ) -> None:
        self.out_shape = out_shape
        self.in_shape = in_shape
        self.initialization_technique = initialization_technique
        # set input shape in Network
        self.bias: Tensor = None
        self.weights: Tensor = None
        
    # overwrite 
    def forward(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):
        for i in range(0, len(in_tensors)):
            out_tensors[i].elements = np.dot(in_tensors[i].elements, self.weights.elements) + self.bias.elements
            #out_tensors[i] = Tensor(elements=out_matrix, deltas=None)
    
    # overwrite
    def backward(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        for i in range(0, len(in_tensors)):
            in_tensors[i].deltas = np.dot(out_tensors[i].deltas, self.weights.elements.T)

    # overwrite
    def calculate_delta_weights(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        # x_matrix = np.array([in_tensor.elements for in_tensor in in_tensors])
        dY_matrix = np.array([out_tensor.deltas for out_tensor in out_tensors])
        # self.weights.deltas = np.divide(np.dot(x_matrix.T, dY_matrix), len(x_matrix))
        # self.bias.deltas = np.divide(np.sum(dY_matrix, axis=0), len(dY_matrix))
        # self.weights.deltas = np.dot(x_matrix.T, dY_matrix)
        self.weights.deltas = np.dot(np.array([in_tensor.elements for in_tensor in in_tensors]).T, dY_matrix)
        self.bias.deltas = np.sum(dY_matrix, axis=0)


        
