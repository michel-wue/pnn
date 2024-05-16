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

    def forward(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):
        for i, tensor in enumerate(in_tensors):
            for x in range(0, len(tensor.elements), self.kernel_size.shape[0]):
                for y in range(0, len(tensor.elements[0]), self.kernel_size.shape[1]):
                    for z in range(self.in_shape.shape[2]):
                        x2 = int(np.divide(x, self.kernel_size.shape[0]))
                        y2 = int(np.divide(y, self.kernel_size.shape[1]))
                        out_tensors[i].elements[x2][y2][z] = np.max([tensor.elements[x + i][y + j][z]
                                                                    for j in range(self.kernel_size.shape[0]) 
                                                                    for i in range(self.kernel_size.shape[1])])
            
    
    def backward(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        for i, tensor in enumerate(in_tensors):
            tensor.deltas = np.zeros(tensor.shape)
            for x in range(0, len(tensor.elements), self.kernel_size.shape[0]):
                for y in range(0, len(tensor.elements[0]), self.kernel_size.shape[1]):
                    for z in range(self.in_shape.shape[2]):
                        maxval = 0
                        max_x = 0
                        max_y = 0
                        for i in range(self.kernel_size.shape[1]):
                            for j in range(self.kernel_size.shape[0]):
                                if tensor.elements[x + i][y + j][z] > maxval:
                                    maxval = tensor.elements[x + i][y + j][z]
                                    max_x = x + i 
                                    max_y = y + j
                        tensor.deltas[max_x][max_y][z] = 1

        

