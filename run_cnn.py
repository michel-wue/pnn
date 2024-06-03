from pnn.layer.fully_connected import FullyConnected
from pnn.layer.convolution import Conv2DLayer
from pnn.layer.pooling import Pooling2DLayer
from pnn.layer.flatten import FlattenLayer
from pnn.layer.activation import ActivationLayer, sigmoid, soft_max, relu
from pnn.layer.loss import LossLayer, mean_squared_error, cross_entropy
from pnn.network import Network
from pnn.trainer import Trainer, sgd
from pnn.shape import Shape
from torchvision.datasets import MNIST
import numpy as np 
from utils import get_plot

if __name__ == "__main__":
    train_data = MNIST(
        root = 'data',
        train = True,                                    
    )
    train_images = np.divide(np.array(train_data.data), np.max(np.array(train_data.data)))
    train_labels = np.array(train_data.targets)

    test_data = MNIST(
        root = 'data', 
        train = False, 
    )
    test_images = np.divide(np.array(test_data.data), np.max(np.array(test_data.data)))
    test_labels = np.array(test_data.targets) 

    conv1 = Conv2DLayer(kernel_size=Shape((3,3)), num_filters=1)
    maxpool1 = Pooling2DLayer(kernel_size=Shape((2,2)), stride=Shape((2,2)), pooling_type='max')
    flatten = FlattenLayer()
    fully_connected_2 = FullyConnected(out_shape=Shape((169,)), initialization_technique='sigmoid')
    activation_layer_sigmoid2 = ActivationLayer(sigmoid)
    fully_connected_4 = FullyConnected(out_shape=Shape((10,)), initialization_technique='softmax') 
    activation_layer_soft_max = ActivationLayer(soft_max)
    loss_layer = LossLayer(cross_entropy)
    layerlist = [conv1,
                maxpool1,
                flatten,
                fully_connected_2, activation_layer_sigmoid2,
                fully_connected_4, activation_layer_soft_max, loss_layer]
    network = Network(layerlist, type = 'convolutional')

    sgd_trainer = Trainer(
        learning_rate=0.03,
        amount_epochs=15,
        update_mechanism=sgd,
        batch_size=1)

    sgd_trainer.optimize(network=network, data=train_images, labels=train_labels)

    prediction = network.predict(test_images)
    avg_time = np.round(np.average(sgd_trainer.parameters['time']))
    prediction_acc = sum(prediction == test_labels)/len(test_labels)
    print(prediction_acc)
    name = 'CNN_sigmoid_cross_entropy'

    get_plot(sgd_trainer=sgd_trainer, name=name, avg_time=avg_time, prediction_acc=prediction_acc)

    network.save_network(f'./networks/{name}')

    conv1 = Conv2DLayer(kernel_size=Shape((3,3)), num_filters=1)
    maxpool1 = Pooling2DLayer(kernel_size=Shape((2,2)), stride=Shape((2,2)), pooling_type='max')
    flatten = FlattenLayer()
    fully_connected_2 = FullyConnected(out_shape=Shape((169,)), initialization_technique='relu')
    activation_layer_sigmoid2 = ActivationLayer(relu)
    fully_connected_4 = FullyConnected(out_shape=Shape((10,)), initialization_technique='softmax') 
    activation_layer_soft_max = ActivationLayer(soft_max)
    loss_layer = LossLayer(cross_entropy)
    layerlist = [conv1,
                maxpool1,
                flatten,
                fully_connected_2, activation_layer_sigmoid2,
                fully_connected_4, activation_layer_soft_max, loss_layer]
    network = Network(layerlist, type = 'convolutional')

    sgd_trainer = Trainer(
        learning_rate=0.03,
        amount_epochs=15,
        update_mechanism=sgd,
        batch_size=1)

    sgd_trainer.optimize(network=network, data=train_images, labels=train_labels)

    prediction = network.predict(test_images)
    avg_time = np.round(np.average(sgd_trainer.parameters['time']))
    prediction_acc = sum(prediction == test_labels)/len(test_labels)
    print(prediction_acc)
    name = 'CNN_relu_cross_entropy'

    get_plot(sgd_trainer=sgd_trainer, name=name, avg_time=avg_time, prediction_acc=prediction_acc)

    network.save_network(f'./networks/{name}')

    conv1 = Conv2DLayer(kernel_size=Shape((3,3)), num_filters=1)
    maxpool1 = Pooling2DLayer(kernel_size=Shape((2,2)), stride=Shape((2,2)), pooling_type='max')
    flatten = FlattenLayer()
    fully_connected_2 = FullyConnected(out_shape=Shape((169,)), initialization_technique='sigmoid')
    activation_layer_sigmoid2 = ActivationLayer(sigmoid)
    fully_connected_4 = FullyConnected(out_shape=Shape((10,)), initialization_technique='softmax') 
    activation_layer_soft_max = ActivationLayer(soft_max)
    loss_layer = LossLayer(mean_squared_error)
    layerlist = [conv1,
                maxpool1,
                flatten,
                fully_connected_2, activation_layer_sigmoid2,
                fully_connected_4, activation_layer_soft_max, loss_layer]
    network = Network(layerlist, type = 'convolutional')

    sgd_trainer = Trainer(
        learning_rate=0.03,
        amount_epochs=15,
        update_mechanism=sgd,
        batch_size=1)

    sgd_trainer.optimize(network=network, data=train_images, labels=train_labels)

    prediction = network.predict(test_images)
    avg_time = np.round(np.average(sgd_trainer.parameters['time']))
    prediction_acc = sum(prediction == test_labels)/len(test_labels)
    print(prediction_acc)
    name = 'CNN_sigmoid_mse'

    get_plot(sgd_trainer=sgd_trainer, name=name, avg_time=avg_time, prediction_acc=prediction_acc)

    network.save_network(f'./networks/{name}')

    conv1 = Conv2DLayer(kernel_size=Shape((3,3)), num_filters=1)
    maxpool1 = Pooling2DLayer(kernel_size=Shape((2,2)), stride=Shape((2,2)), pooling_type='max')
    flatten = FlattenLayer()
    fully_connected_2 = FullyConnected(out_shape=Shape((169,)), initialization_technique='relu')
    activation_layer_sigmoid2 = ActivationLayer(relu)
    fully_connected_4 = FullyConnected(out_shape=Shape((10,)), initialization_technique='softmax') 
    activation_layer_soft_max = ActivationLayer(soft_max)
    loss_layer = LossLayer(mean_squared_error)
    layerlist = [conv1,
                maxpool1,
                flatten,
                fully_connected_2, activation_layer_sigmoid2,
                fully_connected_4, activation_layer_soft_max, loss_layer]
    network = Network(layerlist, type = 'convolutional')

    sgd_trainer = Trainer(
        learning_rate=0.03,
        amount_epochs=15,
        update_mechanism=sgd,
        batch_size=1)

    sgd_trainer.optimize(network=network, data=train_images, labels=train_labels)

    prediction = network.predict(test_images)
    avg_time = np.round(np.average(sgd_trainer.parameters['time']))
    prediction_acc = sum(prediction == test_labels)/len(test_labels)
    print(prediction_acc)
    name = 'CNN_relu_mse'

    get_plot(sgd_trainer=sgd_trainer, name=name, avg_time=avg_time, prediction_acc=prediction_acc)

    network.save_network(f'./networks/{name}')