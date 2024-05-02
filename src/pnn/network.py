import numpy as np

from .tensor import Tensor
from .layer.activation import ActivationLayer
from .layer.fully_connected import FullyConnected
from .layer.input import InputLayer
from .layer.layer import Layer
from .shape import Shape

class Network():
    def __init__(
            self, 
            layers: list[Layer]
            ):
        self.input =  InputLayer()
        self.layers = layers
        self.tensorlist: list[list[Tensor]] = []
        self.initialize: bool = True

    def forward(self, data: list[np.ndarray]) -> Tensor:
        input_tensor = self.input.forward(data)
        length_input = len(input_tensor)
        if self.initialize:
            self.tensorlist.append(input_tensor)
            out_shape: Shape = None
            for layer in self.layers:
                if isinstance(layer, FullyConnected):    
                    out_shape = layer.out_shape.shape
                    layer.in_shape = Shape((len(self.tensorlist[-1][0].elements), 1))
                    layer.bias = Tensor(np.random.rand(out_shape[0]), None)
                    layer.weights = Tensor(np.random.rand(layer.in_shape.shape[0], out_shape[0]), None)   
                    self.tensorlist.append(np.array([Tensor(np.random.rand(out_shape[0]), None) for j in range(0, length_input)]))
                layer.forward(self.tensorlist[-2], self.tensorlist[-1])
            self.initialize = False
        else:
            for i, layer in enumerate(self.layers):
                if not isinstance(layer, InputLayer):
                    layer.forward(self.tensorlist[i-1], self.tensorlist[i])
        return self.tensorlist[-1]
            
    
    def backprop(self):
        for i, layer in reversed(list(enumerate(self.layers))):
            if not isinstance(layer, InputLayer):
                layer.backward(self.tensorlist[i], self.tensorlist[i-1])
                if isinstance(layer, FullyConnected):
                    layer.calculate_delta_weights(self.tensorlist[i], self.tensorlist[i-1])

    def save_params():
        pass

    def load_params():
        pass