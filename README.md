# A Neural Network Library From Scratch
A neural network library I wrote as a component of my undergraduate thesis. The only external library I use to create the neural network is NumPy. I based my library off of Michael Nielsen's [network2.py](https://github.com/mnielsen/neural-networks-and-deep-learning/tree/master/src) neural network library.

# Overview
## The library 
### network_library.py
Contains the code for the neural network library is held.

## Tests
Every test is designed to compare two networks with slightly different settings. Once the program has finished running, a plot showing the performance of both networks on the MNIST dataset will be shown.
### architecture_test.py
Compares two network architectures: 784-100-100-10 vs. 784-100-10.

### cost_function_test.py
Compares a network using the cross-entropy cost function to one using the quadratic cost function

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

## Miscelaneous
### mnist_loader.py
Loads the mnist.pkl.gz file as data 

### mnist.pkl.gz
The MNIST dataset.

### network.txt 
A trained network saved as a textfile, which achieves an accuracy rating of 98.06% on the MNIST test data.
