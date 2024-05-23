import numpy as np

from ..shape import Shape
from ..tensor import Tensor


class Pooling2DLayer:
    def __init__(
            self,
            pooling_type: str,
            kernel_size: Shape,
            stride: Shape,
            in_shape: Shape,
            out_shape: Shape,

    ):
        self.pooling_type = pooling_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.masks = []

    def forward(
            self,
            input_tensor: list[Tensor],
            output_tensor: list[Tensor]
    ):
        elements = []
        for tensor in range(len(input_tensor)):
            mask = []
            for x in range(0, len(input_tensor[tensor].elements), self.stride.shape[0]):
                p = []
                for y in range(0, len(input_tensor[tensor].elements[x]), self.stride.shape[1]):
                    o = []
                    for z in range(0, len(input_tensor[tensor].elements[x][y])):
                        kernel = input_tensor[tensor].elements[
                                 x:x + self.kernel_size.shape[0],
                                 y:y + self.kernel_size.shape[1],
                                 z]
                        value_max = np.max(kernel)
                        o.append(value_max)
                        index = np.where(kernel == value_max)
                        mask.append([x+index[0][0], y+index[1][0], z])
                    p.append(o)
                elements.append(p)
            output_tensor[tensor].elements = elements
            self.masks.append(mask)

    def backward(
            self,
            input_tensor: list[Tensor],
            output_tensor: list[Tensor]
    ):
        for tensor in range(len(input_tensor)):
            output_tensor[tensor].deltas = np.zeros(shape=self.in_shape.shape)
            mask = self.masks[tensor]
            flatt_deltas = input_tensor[tensor].deltas.flatten().tolist()
            for delta in range(len(flatt_deltas)):
                output_tensor[tensor].deltas[*mask[delta]] = flatt_deltas[delta]
            self.masks = []
