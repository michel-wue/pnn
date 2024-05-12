from pnn.layer.input import InputLayer
from pnn.tensor import Tensor
from pnn.layer.fully_connected import FullyConnected
from pnn.layer.convolution import Conv2DLayer
from pnn.layer.activation import ActivationLayer, sigmoid, soft_max, relu
from pnn.layer.loss import LossLayer, mean_squared_error, cross_entropy
from pnn.network import Network
from pnn.trainer import Trainer, sgd
from pnn.shape import Shape
import matplotlib.pyplot as plt
import mnist
import numpy as np 

def get_plot(sgd_trainer, name, avg_time, prediction_acc):
    plt.plot(sgd_trainer.parameters['loss'])
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.text(max(sgd_trainer.parameters['epoche']), max(sgd_trainer.parameters['loss']), f'avg_training_time: {avg_time} \n accuracy: {prediction_acc}', fontsize=12, color='black', ha='right', va='top')
    plt.title(name) 
    plt.savefig(f'./figures/{name}')
    plt.show()

if __name__ == "__main__":
    train_images = mnist.train_images()
    train_images = np.divide(train_images, np.max(train_images))
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_images = np.divide(test_images, np.max(test_images))
    test_labels = mnist.test_labels()

    # # fully_connected_1 = FullyConnected(out_shape=Shape((784,)), initialization_technique='sigmoid')
    # # activation_layer_sigmoid1 = ActivationLayer(sigmoid)
    # fully_connected_2 = FullyConnected(out_shape=Shape((196,)), initialization_technique='sigmoid')
    # activation_layer_sigmoid2 = ActivationLayer(sigmoid)
    # # fully_connected_3 = FullyConnected(out_shape=Shape((49,)), initialization_technique='sigmoid')
    # # activation_layer_sigmoid3 = ActivationLayer(sigmoid)
    # fully_connected_4 = FullyConnected(out_shape=Shape((10,)), initialization_technique='softmax') 
    # activation_layer_soft_max = ActivationLayer(soft_max)
    # loss_layer_cross_entropy = LossLayer(cross_entropy)
    # layerlist = [#fully_connected_1, activation_layer_sigmoid1, 
    #             fully_connected_2, activation_layer_sigmoid2,
    #             #  fully_connected_3, activation_layer_sigmoid3,
    #             fully_connected_4, activation_layer_soft_max, loss_layer_cross_entropy]
    # network = Network(layerlist)

    # sgd_trainer = Trainer(
    #     learning_rate=0.03,
    #     amount_epochs=15,
    #     update_mechanism=sgd,
    #     batch_size=1)
    
    # sgd_trainer.optimize(network=network, data=train_images, labels=train_labels)

    # prediction = network.predict(test_images)
    # avg_time = np.round(np.average(sgd_trainer.parameters['time']))
    # prediction_acc = sum(prediction == test_labels)/len(test_labels)
    # print(prediction_acc)
    # name = 'NN_sigmoid_cross_entropy'

    # get_plot(sgd_trainer=sgd_trainer, name=name, avg_time=avg_time, prediction_acc=prediction_acc)

    # network.save_network(f'./networks/{name}')
    # n = Network.load_network(f'./networks/{name}')

    # prediction = n.predict(test_images)
    # print(f'Prediction accuracy after reloading the network: {sum(prediction == test_labels)/len(test_labels)}')

    # fully_connected_2 = FullyConnected(out_shape=Shape((196,)), initialization_technique='sigmoid')
    # activation_layer_relu = ActivationLayer(relu)
    # fully_connected_4 = FullyConnected(out_shape=Shape((10,)), initialization_technique='softmax') 
    # activation_layer_soft_max = ActivationLayer(soft_max)
    # loss_layer_cross_entropy = LossLayer(cross_entropy)
    # layerlist = [fully_connected_2, activation_layer_relu,
    #             fully_connected_4, activation_layer_soft_max, loss_layer_cross_entropy]
    # network = Network(layerlist)

    # sgd_trainer = Trainer(
    #     learning_rate=0.03,
    #     amount_epochs=15,
    #     update_mechanism=sgd,
    #     batch_size=1)
    
    # sgd_trainer.optimize(network=network, data=train_images, labels=train_labels)

    # prediction = network.predict(test_images)
    # avg_time = np.round(np.average(sgd_trainer.parameters['time']))
    # prediction_acc = sum(prediction == test_labels)/len(test_labels)
    # print(prediction_acc)
    # name = 'NN_relu_cross_entropy'

    # get_plot(sgd_trainer=sgd_trainer, name=name, avg_time=avg_time, prediction_acc=prediction_acc)

    # network.save_network(f'./networks/{name}')
  
    fully_connected_2 = FullyConnected(out_shape=Shape((196,)), initialization_technique='sigmoid')
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