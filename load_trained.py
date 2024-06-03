from pnn.network import Network
from torchvision.datasets import MNIST
import numpy as np

if __name__ == '__main__':
    test_data = MNIST(
        root = 'data', 
        train = False, 
    )
    test_images = np.divide(np.array(test_data.data), np.max(np.array(test_data.data)))
    test_labels = np.array(test_data.targets)
    cnn = Network.load_network('./networks/CNN_relu_cross_entropy')

    prediction = cnn.predict(test_images)
    print(f'Prediction accuracy after reloading the network: {sum(prediction == test_labels)/len(test_labels)}')

    nn = Network.load_network('./networks/NN_sigmoid_cross_entropy')

    prediction = nn.predict(test_images)
    print(f'Prediction accuracy after reloading the network: {sum(prediction == test_labels)/len(test_labels)}')