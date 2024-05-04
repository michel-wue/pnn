import numpy as np

from .tensor import Tensor
from .layer.activation import ActivationLayer
from .layer.fully_connected import FullyConnected
from .layer.loss import LossLayer
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
        self.labels: list[Tensor]

    def _transform_labels(self, labels: np.ndarray, unique_length: int) -> list[Tensor]:
        label_list = [Tensor(np.zeros(shape=(unique_length,))) for i in range(len(labels))]
        for i, label in enumerate(labels):
            np.put(label_list[i].elements, label, 1)
        return label_list

    def forward(self, data: list[np.ndarray], labels: np.ndarray, unique_length: int = 0) -> np.float64:
        if self.initialize:
            self.labels = self._transform_labels(labels, unique_length=unique_length)
            input_tensor = self.input.forward(data)
            length_input = len(input_tensor)
            self.tensorlist.append(input_tensor)
            out_shape: Shape = None
            for layer in self.layers:
                if isinstance(layer, FullyConnected):    
                    out_shape = layer.out_shape.shape
                    layer.in_shape = Shape((len(self.tensorlist[-1][0].elements), 1))
                    layer.bias = Tensor(np.random.rand(out_shape[0]), None)
                    layer.weights = Tensor(_init_weightmatrix((layer.in_shape.shape[0], out_shape[0]), layer.initialization_technique), None)
                    # layer.weights = Tensor(np.random.rand(layer.in_shape.shape[0], out_shape[0]), None)
                if not isinstance(layer, LossLayer):
                    self.tensorlist.append(np.array([Tensor(np.random.rand(out_shape[0]), None) for j in range(0, length_input)]))
                    layer.forward(self.tensorlist[-2], self.tensorlist[-1])    
            self.initialize = False
        else:
            for i in range(len(self.layers)-1):
                if i == 0:
                    self.tensorlist[0] = self.input.forward(data)
                self.layers[i].forward(self.tensorlist[i], self.tensorlist[i+1])
        # calculate loss with last element from tensorlist + labels
        return self.layers[-1].forward(self.tensorlist[-1], self.labels)
            
    
    def backprop(self):
        for i, layer in reversed(list(enumerate(self.layers))):
            if isinstance(layer, LossLayer):
                layer.backward(predictions=self.tensorlist[i], targets=self.labels)
            else:
            # backward weglassen bei i = 0? deltas werden glaube nicht mehr benötigt
                layer.backward(out_tensors=self.tensorlist[i+1], in_tensors=self.tensorlist[i])
                if isinstance(layer, FullyConnected):
                    # print(1)
                    layer.calculate_delta_weights(out_tensors=self.tensorlist[i+1], in_tensors=self.tensorlist[i])

    def save_params():
        pass

    def load_params():
        pass

def _init_weightmatrix(shape: tuple, initialization: str) -> np.ndarray:
    # He initialization 
    if initialization == 'relu':
        limit = np.sqrt(2 / shape[0])
        return np.random.uniform(-limit, limit, size=shape)
    # Xavier initilization
    elif initialization == 'tanh' or initialization == 'sigmoid' or initialization == 'softmax':
        limit = np.sqrt(6 / np.sum(shape))
        return np.random.uniform(-limit, limit, size=shape)
    else:
        raise ValueError('{initialization} not one of: relu, sigmoid, tanh, softmax'.format(shape=repr(initialization)))