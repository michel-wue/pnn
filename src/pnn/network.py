import numpy as np

from .tensor import Tensor
from .layer.activation import ActivationLayer
from .layer.fully_connected import FullyConnected
from .layer.input import InputLayer
from .layer.layer import Layer

class Network():
    def __init__(
            self,
            layers: list[Layer]
            ):
        self.input =  InputLayer()
        self.layers = layers
        self.tensorlist = []

    def forward(self, data):
        input_tensor = self.input.forward(data)
        length_input = len(input_tensor)
        self.tensorlist.append(input_tensor)
        out_shape = None
        for layer in self.layers:
            if isinstance(layer, FullyConnected):    
                out_shape = layer.out_shape
                layer.bias = Tensor(np.random.rand(out_shape[0]), None)
                layer.weightmatrix = Tensor(np.random.rand(out_shape[0], layer.in_shape[0]), None)   
            self.tensorlist.append(np.array([Tensor(np.random.rand(out_shape[0]), None)
                                                for j in range(0, length_input)]))
            layer.forward(self.tensorlist[-2], self.tensorlist[-1])

        # prediction = np.array([self.tensorlist[-1][i].elements.argmax(axis=0) for i in range(len(self.tensorlist[-1]))])
        return self.tensorlist[-1]
            
    
    def backprop():
        pass

    def save_params():
        pass

    def load_params():
        pass