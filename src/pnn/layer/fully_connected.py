from .layer import Layer
from ..tensor import Tensor
from ..shape import Shape
import numpy as np

class FullyConnected(Layer):
    def __init__(
            self, 
            out_shape: tuple,
            ) -> None:
        self.out_shape = out_shape
        # set input shape in Network
        self.in_shape = None
        self.bias = None
        self.weightmatrix = None
        
    # overwrite 
    def forward(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):
        self.in_shape = (len(in_tensors[0].elements),)
        self.bias = Tensor(np.random.rand(self.out_shape[0]), None)
        self.weightmatrix = Tensor(np.random.rand(self.out_shape[0], self.in_shape[0]), None)
        for i in range(0, len(in_tensors)):
            out_matrix = np.dot(self.weightmatrix.elements, 
                                in_tensors[i].elements) + self.bias.elements
            out_matrix = np.array(out_matrix)
            out_tensors[i] = Tensor(elements=out_matrix, deltas=None)
    
    # overwrite
    def backward(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        pass

    # overwrite
    def calculate_delta_weights(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        pass

    def save_params():
        pass

    def load_params():
        pass
