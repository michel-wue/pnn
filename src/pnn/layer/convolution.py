import numpy as np
from ..shape import Shape
from .layer import Layer
from ..tensor import Tensor

class Conv2DLayer(Layer):
    def __init__(
            self,
            out_shape: Shape,
            kernel_size: Shape,
            num_filters: int,
            stride: tuple = None,
            dilation: tuple = None,
            padding: str = None,
            in_shape: Shape = None,
        ):
        self.out_shape = out_shape
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.in_shape = in_shape
        self.weights = Tensor(np.random.rand(kernel_size.shape[0], kernel_size.shape[1], num_filters, in_shape.shape[2]), None)
        self.bias = Tensor(np.zeros(num_filters), None)

    def forward(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):
        for i, tensor in enumerate(in_tensors):
            for y in range(len(tensor.elements) - self.kernel_size.shape[0] + 1):
                for x in range(len(tensor.elements[0]) - self.kernel_size.shape[1] + 1):
                    for z in range(self.num_filters):
                        out_tensors[i].elements[y][x][z] = np.sum([np.multiply(tensor.elements[y + i][x + j][a], self.weights.elements[i][j][a][z]) for j in range(self.kernel_size.shape[1]) for i in range(self.kernel_size.shape[0]) for a in range(self.in_shape.shape[2])])


    def backward(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        pass