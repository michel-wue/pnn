from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, input):
        pass

    @abstractmethod
    def calculate_delta_weights(self, input):
        pass
