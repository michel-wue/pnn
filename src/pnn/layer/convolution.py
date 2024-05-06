from ..shape import Shape
from .layer import Layer
from ..tensor import Tensor

class Conv2DLayer(Layer):
    def __init__(
            self,
            out_shape: Shape,
            kernel_size: Shape,
            num_filters: int,
            stride: tuple,
            dilation: tuple,
            padding: str,
            in_shape: Shape = None,
        ):
        self.out_shape = out_shape
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.in_shape = in_shape
        self.filter = None

    def _init_filter(self):
        

    def forward(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):


    def backward(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        pass