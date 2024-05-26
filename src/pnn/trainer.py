import numpy as np
# from .layer.fully_connected import FullyConnected
from typing import Callable
from .network import Network, FullyConnected
import random
import time
from tqdm import tqdm

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
        self.parameters = {'epoche': [], 'loss': [], 'time': []}

    def optimize(self, network: Network, data: list[np.ndarray], labels: np.ndarray) -> None:
        # unique_length = len(np.unique(labels)) # required for label init
        unique_length = 10 # required for label init
        for i in range(self.amount_epochs):
            start = time.time()
            epoch_loss = []
            if self.shuffle:
                data, labels = _shuffle(data=data, labels=labels)
            number_of_batches = int(np.ceil(len(data)/self.batch_size))
            unequal_size = len(data)%self.batch_size
            unequal_size_bool: bool = unequal_size == 0
            b = []
            for j in tqdm(range(number_of_batches)):
                batch = []
                batch_labels = []
                if j == number_of_batches - 1 and not unequal_size_bool:
                    batch = data[j*self.batch_size : (j)*self.batch_size + unequal_size]
                    batch_labels = labels[j*self.batch_size : (j)*self.batch_size + unequal_size]
                else:    
                    batch = data[j*self.batch_size:(j+1)*self.batch_size]
                    batch_labels = labels[j*self.batch_size:(j+1)*self.batch_size]
                batch_labels = np.array(batch_labels)
                epoch_loss.append(network.forward(data=batch, labels=batch_labels, unique_length=unique_length))
                network.backprop()
                self.update_mechanism(network=network, learning_rate=self.learning_rate)
            end = time.time()
            self.parameters['epoche'].append(i)
            self.parameters['loss'].append(round(np.average(epoch_loss), 4))
            self.parameters['time'].append(round(end-start, 4))
            print(f"epoche: {i}, loss: {self.parameters['loss'][i]}, time: {self.parameters['time'][i]}s")

            
def sgd(network: Network, learning_rate: float) -> None:
    for layer in network.layers:
        if isinstance(layer, FullyConnected):
            layer.weights.elements = layer.weights.elements - np.multiply(learning_rate, layer.weights.deltas)
            layer.bias.elements = layer.bias.elements - np.multiply(learning_rate, layer.bias.deltas)

def _shuffle(data: list[np.ndarray], labels: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    c = list(zip(data, labels))
    random.shuffle(c)
    data, labels = zip(*c)
    return list(data), np.array(labels)