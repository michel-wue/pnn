from src.pnn.layer.input import InputLayer
from src.pnn.tensor import Tensor
from src.pnn.layer.fully_connected import FullyConnected
from src.pnn.layer.activation import ActivationLayer, sigmoid, soft_max
from src.pnn.layer.loss import LossLayer, mean_squared_error, cross_entropy
from src.pnn.network import Network
from src.pnn.shape import Shape
import mnist

if __name__ == "__main__":
    mnist.temporary_dir = lambda: './data'
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    fully_connected_1 = FullyConnected(out_shape=Shape((100,)))
    activation_layer_sigmoid = ActivationLayer(sigmoid)
    fully_connected_2 = FullyConnected(out_shape=Shape((10,))) 
    activation_layer_soft_max = ActivationLayer(soft_max)
    loss_layer = LossLayer(mean_squared_error)
    layerlist = [fully_connected_1, activation_layer_sigmoid, fully_connected_2, activation_layer_soft_max, loss_layer]
    network = Network(layerlist)
    network.forward(train_images, train_labels)
    network.forward(train_images, train_labels)
    network.backprop()
