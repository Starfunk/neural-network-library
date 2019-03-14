"""Using this network setup, I have achieved a maximum accuracy
of 98.06% on the MNIST dataset."""

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

# Instantiate the neural network with an architecture of 784-100-100-10,
# a learning rate of 0.6, a minibatch size of 64, using the 
# cross-entropy function and small weight initialization.
net = network_library.Network([784, 100, 100, 10], 0.6
, 64, cost_function=network_library.CrossEntropyCost, small_weights=True)
	
# Run mini-batch stochastic gradient descent for 60,000 training epochs
# with L2 regularization.
net.stochastic_gradient_descent(training_data, training_labels, 
	60000, L2=True)

# Evaluate the network's performance on the test dataset.	
acc = net.evaluation(test_data, test_labels)

print("The network classified " + str(acc) + "% of the test data")

# If you would like to verify that I trained a network that achieved 
# an accuracy of 98.06%, delete the quotations in the following section.

"""
net = network_library.load('network.txt')
acc = net.evaluation(test_data, test_labels)
print("The network classified " + str(acc) + "% of the test data")
"""
