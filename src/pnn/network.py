import numpy as np

from .tensor import Tensor
from .layerivationLayer
from .layer.fully_connected import FullyConnected
from .layer.input_layer import InputLayer
from .layer.layer import Layer

class Network():
    def __init__(
            self,
            input: InputLayer,
            layers: list[Layer],
            ):
        self.input = input
        self.layers = layers

    def forward(self):
        for layer in self.layers:
            out_shape = None
            if isinstance(layer, InputLayer):
                self.tensorlist = list(layer.forward())
            elif isinstance(layer, FullyConnected):
                out_shape = layer.out_shape
                self.tensorlist.append(np.array([Tensor(np.random.rand(out_shape[0]), None)
                                                  for j in range(0, len(self.tensorlist[-1]))]))
                layer.forward(self.tensorlist[-2], self.tensorlist[-1])
            elif isinstance(layer, ActivationLayer):
                self.tensorlist(np.array([Tensor(np.random.rand(out_shape[0]), None) 
                                          for j in range(0, len(self.tensorlist[-1]))]))
                layer.forward(self.tensorlist[-2], self.tensorlist[-1])

        prediction = np.array([self.tensorlist[-1][i].elements.argmax(axis=0) for i in range(len(self.tensorlist[-1]))])
        return prediction
            
    
    def backprop():
        pass

    def save_params():
        pass

    def load_params():
        pass