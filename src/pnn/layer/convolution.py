import numpy as np
from ..shape import Shape
from .layer import Layer
from ..tensor import Tensor
from tqdm import tqdm


class Conv2DLayer(Layer):
    def __init__(
            self,
            kernel_size: Shape,
            num_filters: int,
            out_shape: Shape = None,
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
        #self.weights = Tensor(np.random.rand(kernel_size.shape[0], kernel_size.shape[1], num_filters, in_shape.shape[2]), None)
        self.weights: Tensor = None
        self.bias = Tensor(np.zeros(num_filters), None)

    def forward(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):
        for i, tensor in enumerate(in_tensors):
            for x in range(len(tensor.elements) - self.kernel_size.shape[0] + 1):
                for y in range(len(tensor.elements[0]) - self.kernel_size.shape[1] + 1):
                    for z in range(self.num_filters):
                        out_tensors[i].elements[x][y][z] = np.sum(np.multiply(
                            tensor.elements[x: x + self.kernel_size.shape[1], y: y + self.kernel_size.shape[0],
                            0: self.in_shape.shape[2]],
                            self.weights.elements[0: self.kernel_size.shape[1], 0: self.kernel_size.shape[0],
                            0: self.in_shape.shape[2], z]))

    def backward(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        # maybe always equals the input shape
        padding_x = self.kernel_size.shape[0] - 1
        padding_y = self.kernel_size.shape[1] - 1
        # output_x = out_shape.shape[0] + 2 * padding_x - (self.kernel_size.shape[0] - 1)
        # output_y = out_shape.shape[1] + 2 * padding_y - (self.kernel_size.shape[1] - 1)
        # padded_shape = (output_x, output_y, out_shape.shape[2])
        # padded_array = np.zeros(padded_shape)

        rotated_filter = np.zeros(shape=self.weights.shape)
        # rotated_filter = self.weights.elements[(self.kernel_size.shape[1]-1)-self.kernel_size.shape[1]: (self.kernel_size.shape[1]-1),
        #                                        (self.kernel_size.shape[0]-1)-self.kernel_size.shape[0]: (self.kernel_size.shape[0]-1),
        #                                        0:self.in_shape.shape[2],
        #                                        0: self.num_filters]
        for z in range(self.num_filters):
            for a in range(self.in_shape.shape[2]):
                for i in range(self.kernel_size.shape[1]):
                    for j in range(self.kernel_size.shape[0]):
                        rotated_filter[i][j][a][z] = \
                        self.weights.elements[(self.kernel_size.shape[1] - 1) - i][(self.kernel_size.shape[0] - 1) - j][
                            a][z]

        for i, in_tensor in enumerate(in_tensors):
            padded_array = np.zeros(self.in_shape.shape)
            padded_array[padding_x: padding_x + self.out_shape.shape[0],
            padding_y: padding_y + self.out_shape.shape[1]] = out_tensors[i].deltas
            in_tensor.deltas = np.zeros(in_tensor.shape)
            for x in range(len(padded_array)):
                for y in range(len(padded_array[0])):
                    for a in range(in_tensor.shape[2]):
                        cut_edge_x = np.maximum(0, x + self.kernel_size.shape[0] - len(padded_array))
                        cut_edge_y = np.maximum(0, y + self.kernel_size.shape[1] - len(padded_array[0]))
                        in_tensor.deltas[x][y][a] = np.sum(
                            np.multiply(padded_array[x:x + self.kernel_size.shape[0], y: y + self.kernel_size.shape[1]],
                                        rotated_filter[0:self.kernel_size.shape[0] - cut_edge_x,
                                        0:self.kernel_size.shape[1] - cut_edge_y, a]))

    def calculate_delta_weights(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        self.weights.deltas = np.zeros(self.weights.shape)
        for tensor_iter, in_tensor in enumerate(in_tensors):
            x = (len(in_tensor.elements) - self.kernel_size.shape[0] + 1)
            y = (len(in_tensor.elements[0]) - self.kernel_size.shape[1] + 1)
            for filter in range(self.num_filters):
                for i in range(self.kernel_size.shape[0]):
                    for j in range(self.kernel_size.shape[1]):
                        for a in range(in_tensor.shape[2]):
                            self.weights.deltas[i][j][a][filter] = np.sum(in_tensor.elements[i: i + x, j: j + y, a] *
                                                                          out_tensors[tensor_iter].deltas[0:x, 0:y,
                                                                          filter])
