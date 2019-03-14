# A Neural Network Library From Scratch
A neural network library written in Python as partial fulfilment of my final undergraduate project. The only external library I use to create the neural network is NumPy. I based my library off of Michael Nielsen's [network2.py](https://github.com/mnielsen/neural-networks-and-deep-learning/tree/master/src) neural network library. The differences between my library and Nielsen's are outlined in network_library.py.

# Overview
### network_library.py
This is the central file of my project. This file represents the "neural network library",i.e., it contains all the code necessary to create and train a neural network.

## Tests
Each test is designed to compare a key neural network feature. For example, we test the performance of a network using L2 regularization to one that is not using L2 regularization.
### architecture_test.py
Compares two network architectures: 784-100-100-10 and 784-100-10.

### cost_function_test.py
Compares a network using the cross-entropy cost function to one using the quadratic cost function.

### small_weight_initialization_test.py
Compares a network using small weight initialization to one not using small weight initialization.

### L2_regularization_test.py
Compares a network using L2 regularization to one not using L2 regularization.

### dropout_test.py
Compares a network using Dropout to one not using Dropout.

### dropconnect_test.py
Compares a network using DropConnect to one not using DropConnect.

### train_network.py
Train the network using the settings I found to achieve the highest results. This file also allows you to load the trained network, 'network.txt', which achieves an accuracy level of 98.06% on the test data.

## Miscellaneous
A set of miscellaneous files that support the function of the neural network library.
### mnist_loader.py
Loads the mnist.pkl.gz file as data.

### mnist.pkl.gz
The MNIST dataset.

### network.txt 
A trained network saved as a textfile, which achieves an accuracy rating of 98.06% on the MNIST test data.
