import numpy as np
# from .layer.fully_connected import FullyConnected
from typing import Callable
from .network import Network, FullyConnected

class Trainer:
    def __init__(
            self,
            learning_rate: float,
            amount_epochs: int,
            update_mechanism: Callable,
            batch_size: int = 1,
            shuffle: bool = True) -> None:
        self.learning_rate = learning_rate
        self.amount_epochs = amount_epochs
        self.update_mechanism = update_mechanism
        self.batch_size = batch_size
        self.shuffle = shuffle

    def optimize(self, network: Network, data: list[np.ndarray], labels: np.ndarray):
        # unique_length = len(np.unique(labels)) # required for label init
        unique_length = 10 # required for label init
        loss = 0
        for i in range(self.amount_epochs):
            number_of_batches = int(np.ceil(len(data)/self.batch_size))
            unequal_size = len(data)%self.batch_size
            unequal_size_bool: bool = unequal_size == 0
            for j in range(number_of_batches):
                batch = []
                batch_labels = []
                if j == number_of_batches - 1 and not unequal_size_bool:
                    batch = data[j*self.batch_size : (j)*self.batch_size + unequal_size]
                    batch_labels = labels[j*self.batch_size : (j)*self.batch_size + unequal_size]
                else:    
                    batch = data[j*self.batch_size:(j+1)*self.batch_size]
                    batch_labels = labels[j*self.batch_size:(j+1)*self.batch_size]
                batch_labels = np.array(batch_labels)
                loss = network.forward(data=batch, labels=batch_labels, unique_length=unique_length)
                network.backprop()
                self.update_mechanism(network=network, learning_rate=self.learning_rate)
            print(f"epoche: {i}, loss: {loss}")
            
def sgd(network: Network, learning_rate: float):
    for layer in network.layers:
        if isinstance(layer, FullyConnected):
            layer.weights.elements = layer.weights.elements - np.multiply(learning_rate, layer.weights.deltas)
            layer.bias.elements = layer.bias.elements - np.multiply(learning_rate, layer.bias.deltas)