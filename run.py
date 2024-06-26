from pnn.layer.fully_connected import FullyConnected
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
    
    # fully_connected_1 = FullyConnected(out_shape=Shape((784,)), initialization_technique='sigmoid')
    # activation_layer_sigmoid1 = ActivationLayer(sigmoid)
    fully_connected_2 = FullyConnected(out_shape=Shape((196,)), initialization_technique='sigmoid')
    activation_layer_sigmoid2 = ActivationLayer(sigmoid)
    # fully_connected_3 = FullyConnected(out_shape=Shape((49,)), initialization_technique='sigmoid')
    # activation_layer_sigmoid3 = ActivationLayer(sigmoid)
    fully_connected_4 = FullyConnected(out_shape=Shape((10,)), initialization_technique='softmax') 
    activation_layer_soft_max = ActivationLayer(soft_max)
    loss_layer_cross_entropy = LossLayer(cross_entropy)
    layerlist = [#fully_connected_1, activation_layer_sigmoid1, 
                fully_connected_2, activation_layer_sigmoid2,
                #  fully_connected_3, activation_layer_sigmoid3,
                fully_connected_4, activation_layer_soft_max, loss_layer_cross_entropy]
    network = Network(layerlist)

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
    name = 'NN_sigmoid_cross_entropy'

    get_plot(sgd_trainer=sgd_trainer, name=name, avg_time=avg_time, prediction_acc=prediction_acc)

    network.save_network(f'./networks/{name}')

    fully_connected_2 = FullyConnected(out_shape=Shape((196,)), initialization_technique='relu')
    activation_layer_relu = ActivationLayer(relu)
    fully_connected_4 = FullyConnected(out_shape=Shape((10,)), initialization_technique='softmax') 
    activation_layer_soft_max = ActivationLayer(soft_max)
    loss_layer_cross_entropy = LossLayer(cross_entropy)
    layerlist = [fully_connected_2, activation_layer_relu,
                fully_connected_4, activation_layer_soft_max, loss_layer_cross_entropy]
    network = Network(layerlist)

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
    name = 'NN_relu_cross_entropy'

    get_plot(sgd_trainer=sgd_trainer, name=name, avg_time=avg_time, prediction_acc=prediction_acc)

    network.save_network(f'./networks/{name}')
  
    fully_connected_2 = FullyConnected(out_shape=Shape((196,)), initialization_technique='relu')
    activation_layer_relu = ActivationLayer(relu)
    fully_connected_4 = FullyConnected(out_shape=Shape((10,)), initialization_technique='softmax') 
    activation_layer_soft_max = ActivationLayer(soft_max)
    loss_layer_mse = LossLayer(mean_squared_error)
    layerlist = [fully_connected_2, activation_layer_relu,
                fully_connected_4, activation_layer_soft_max, loss_layer_mse]
    network = Network(layerlist)

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
    name = 'NN_relu_mse'

    get_plot(sgd_trainer=sgd_trainer, name=name, avg_time=avg_time, prediction_acc=prediction_acc)

    network.save_network(f'./networks/{name}')

    fully_connected_2 = FullyConnected(out_shape=Shape((196,)), initialization_technique='sigmoid')
    activation_layer_sigmoid = ActivationLayer(sigmoid)
    fully_connected_4 = FullyConnected(out_shape=Shape((10,)), initialization_technique='softmax') 
    activation_layer_soft_max = ActivationLayer(soft_max)
    loss_layer_mse = LossLayer(mean_squared_error)
    layerlist = [fully_connected_2, activation_layer_sigmoid,
                fully_connected_4, activation_layer_soft_max, loss_layer_mse]
    network = Network(layerlist)

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
    name = 'NN_sigmoid_mse'

    get_plot(sgd_trainer=sgd_trainer, name=name, avg_time=avg_time, prediction_acc=prediction_acc)

    network.save_network(f'./networks/{name}')