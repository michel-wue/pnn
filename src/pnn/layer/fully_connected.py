from .layer import Layer
from ..tensor import Tensor
from ..shape import Shape
import numpy as np

class FullyConnected(Layer):
    def __init__(
            self, 
            # weightmatrix: Tensor,
            # bias: Tensor
            in_shape: tuple,
            out_shape: tuple
            ) -> None:
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.bias = Tensor(np.random.rand(out_shape[0]), None)
        self.weightmatrix = Tensor(np.random.rand(out_shape[0], in_shape[0]), None)
        # self.in_shape = in_shape
        # self.out_shape = out_shape
        
    # overwrite 
    def forward(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):
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
