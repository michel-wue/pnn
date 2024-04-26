import numpy as np

from pnn.tensor import Tensor
from .layer import Layer

class SoftmaxLayer(Layer):
    def forward(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):
        pass
    
    def backward(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        pass