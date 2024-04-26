from .layer import Layer
from ..tensor import Tensor
from ..shape import Shape
import numpy as np

class FullyConnected(Layer):
    def __init__(
            self, 
            weightmatrix: Tensor,
            bias: Tensor
            # in_shape: Shape,
            # out_shape:Shape
            ) -> None:
        self.weightmatrix = weightmatrix
        self.bias = bias
        # self.in_shape = in_shape
        # self.out_shape = out_shape
        
    # overwrite 
    def forward(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):
        for i in range(0, len(in_tensors)):
            out_tensors[i] = in_tensors[i].elements * self.weightmatrix.elements + self.bias
    
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
