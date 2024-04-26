from abc import ABC, abstractmethod
from ..tensor import Tensor

class Layer(ABC):
    @abstractmethod
    def forward(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):
        pass

    @abstractmethod
    def backward(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        pass

    def calculate_delta_weights(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        pass
