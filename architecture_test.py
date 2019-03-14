"""Investigating the effect network architecture has on MNIST 
performance."""

import numpy as np
import network_library
import mnist_loader
				
# load the MNIST dataset using the mnist_loader file---the returned
# data structure is a list containing the training data, the validation
# data, and the test data.
data = mnist_loader.load_mnist()

# Set training data and training labels. The training dataset is a set 
# of 50,000 MNIST (hand-drawn digit) images.
training_data = data[0]
training_labels = data[1]

# Set validation data and validation labels. The validation dataset is a
# set of 10,000 MNIST (hand-drawn digit) images.
validation_data = data[2]
validation_labels = data[3]

# Set test data and test labels. The test dataset is a
# set of 10,000 MNIST (hand-drawn digit) images.
test_data = data[4]
test_labels = data[5]

# Instantiate the first neural network.
net1 = network_library.Network([784, 100, 100, 10], 0.6, 64, 
	cost_function=network_library.QuadraticCost)
	
# Instantiate the second neural network.
net2 = network_library.Network([784, 100, 10], 0.6, 64, 
	cost_function=network_library.QuadraticCost)
	
# Compare both networks.
network_library.compare_net(net1, net2, training_data, training_labels,
	test_data, test_labels, 1000, 64, 'Four-layer network', 
	'Three-layer network')
