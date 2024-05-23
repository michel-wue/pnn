import numpy as np
import os

from .tensor import Tensor
from .layer.activation import ActivationLayer
from .layer.fully_connected import FullyConnected
from .layer.loss import LossLayer
from .layer.input import InputLayer
from .layer.layer import Layer
from .layer.convolution import Conv2DLayer
from .layer.pooling import Pooling2DLayer
from .layer.flatten import FlattenLayer
from .shape import Shape
import pickle
from tqdm import tqdm

class Network():
    def __init__(
            self, 
            layers: list[Layer],
            type: str = 'fully_connected'
            ):
        self.input =  InputLayer(type)
        self.layers = layers
        self.tensorlist: list[list[Tensor]] = []
        self.initialize: bool = True
        self.labels: list[Tensor]
        self.t = None

    def _transform_labels(self, labels: np.ndarray, unique_length: int) -> list[Tensor]:
        label_list = [Tensor(np.zeros(shape=(unique_length,))) for i in range(len(labels))]
        for i, label in enumerate(labels):
            np.put(label_list[i].elements, label, 1)
        return label_list

    def forward(self, data: list[np.ndarray], labels: np.ndarray, unique_length: int = 10) -> np.float64:
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
                elif isinstance(layer, Conv2DLayer):
                    layer.in_shape = Shape((self.tensorlist[-1][0].shape))
                    # print(layer.in_shape)
                    layer.out_shape = Shape((layer.in_shape.shape[0] - layer.kernel_size.shape[0] + 1, 
                                       layer.in_shape.shape[1]  - layer.kernel_size.shape[1] + 1, 
                                       layer.num_filters))
                    out_shape = layer.out_shape.shape
                    layer.bias = Tensor(np.zeros(layer.num_filters), None)
                    layer.weights = Tensor(_init_weightmatrix((layer.kernel_size.shape[0], layer.kernel_size.shape[1], layer.in_shape.shape[2], layer.num_filters), 'convolution'), None)
                elif isinstance(layer, FlattenLayer):
                    out_shape = np.ndarray.flatten(self.tensorlist[-1][0].elements).shape 

                # Append to tensorlist
                if not isinstance(layer, LossLayer):
                    #print(layer.__class__)
                    self.tensorlist.append(np.array([Tensor(np.zeros(out_shape), None) for j in range(0, length_input)]))
                    layer.forward(in_tensors=self.tensorlist[-2], out_tensors=self.tensorlist[-1])    
                    # print(self.tensorlist[-1][0].elements.shape)
            self.initialize = False
        else:
            for i in range(len(self.layers)-1):
                if i == 0:
                # if isinstance(self.layers[i], InputLayer): # does not work because isinstance does not recognize input layer
                    self.tensorlist[i] = self.input.forward(data)
                    self.labels = self._transform_labels(labels, unique_length=unique_length)
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
                    layer.calculate_delta_weights(out_tensors=self.tensorlist[i+1], in_tensors=self.tensorlist[i])

    # Shapes will likely be problematic when saving the network

    def predict(self, data: list[np.ndarray]) -> list[int]:
        # last layer is loss layer so it gets cut out in predict
        # prediction = []
        # for input in data:
        #     for i in range(len(self.layers)-1):
        #         if i == 0:
        #             self.tensorlist[0] = self.input.forward([input])
        #         self.layers[i].forward(self.tensorlist[i], self.tensorlist[i+1])

        #     single_prediciton = np.argmax(self.tensorlist[-1][0].elements)
        #     prediction.append(single_prediciton)
        #     print(single_prediciton)
        # return np.array(prediction)

        length_input = len(data)
        for i in range(len(self.layers)-1):
            out_shape = self.tensorlist[i+1][0].shape
            # change size of output tensor to match required size ()
            self.tensorlist[i+1] = np.array([Tensor(np.zeros(out_shape), None) for j in range(length_input)])
            if i == 0:
                self.tensorlist[0] = self.input.forward(data)
            self.layers[i].forward(self.tensorlist[i], self.tensorlist[i+1])

        return np.array([np.argmax(tensor.elements) for tensor in self.tensorlist[-1]])

    def save_network(self, filename):
        if not os.path.isdir('networks'):
            os.mkdir('networks')
        pickle.dump(self, open(filename, 'wb'))
        
    @classmethod
    def load_network(cls, filename):
        return pickle.load(open(filename, 'rb'))

def _init_weightmatrix(shape: tuple, initialization: str) -> np.ndarray:
    # He initialization 
    if initialization == 'relu' or initialization == 'convolution':
        limit = np.sqrt(2 / shape[0])
        return np.random.uniform(-limit, limit, size=shape)
    # Xavier initilization
    elif initialization == 'tanh' or initialization == 'sigmoid' or initialization == 'softmax':
        limit = np.sqrt(6 / np.sum(shape))
        return np.random.uniform(-limit, limit, size=shape)
    else:
        raise ValueError('{initialization} not one of: relu, sigmoid, tanh, softmax'.format(shape=repr(initialization)))