from .layer import Layer
from ..tensor import Tensor
from ..shape import Shape
import numpy as np

class FlattenLayer(Layer):
    def forward(self, in_tensors: list[Tensor], out_tensors: list[Tensor]) -> None:
        for i, tensor in enumerate(in_tensors):
            out_tensors[i].elements = np.ndarray.flatten(tensor.elements, order='C')
    
    def backward(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        for i, tensor in enumerate(in_tensors):
            tensor.deltas = out_tensors[i].deltas.reshape(out_tensors[i].shape)