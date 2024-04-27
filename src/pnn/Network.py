from .layer.activation import activation
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
        
    def backprop():
        pass

    def forward():
        pass

    def save_params():
        pass

    def load_params():
        pass