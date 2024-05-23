from ..tensor import Tensor
from ..shape import Shape
import numpy as np
from .layer import Layer

class Pooling2DLayer(Layer):
    def __init__(
            self,
            kernel_size: Shape, 
            stride: Shape, 
            pooling_type: str, 
            in_shape: Shape, 
            out_shape: Shape) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.pooling_type = pooling_type
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.masks = []

    def forward(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):
        for i, tensor in enumerate(in_tensors):
            mask = []
            for x in range(0, len(tensor.elements), self.kernel_size.shape[0]):
                for y in range(0, len(tensor.elements[0]), self.kernel_size.shape[1]):
                    for z in range(self.in_shape.shape[2]):
                        x2 = int(np.divide(x, self.kernel_size.shape[0]))
                        y2 = int(np.divide(y, self.kernel_size.shape[1]))
                        kernel = tensor.elements[
                                 x:x + self.kernel_size.shape[0],
                                 y:y + self.kernel_size.shape[1],
                                 z]
                        out_tensors[i].elements[x2][y2][z] = np.max(kernel)
                        index = np.where(kernel == out_tensors[i].elements[x2][y2][z])
                        mask.append([x+index[0][0], y+index[1][0], z])
            self.masks.append(mask)


    
    def backward(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        for i, tensor in enumerate(in_tensors):
            tensor.deltas = np.zeros(shape=self.in_shape.shape)
            mask = self.masks[i]
            flatt_deltas = out_tensors[i].deltas.flatten().tolist()
            for delta in range(len(flatt_deltas)):
                tensor.deltas[*mask[delta]] = flatt_deltas[delta]
            self.masks = []

        

