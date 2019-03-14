"""The MNIST dataset is taken from Michael Nielsen's Neural Networks 
and Deep Learning repository. The file is adapted from 
Nielsen's mnist_loader.py file found in the same repository:
https://github.com/mnielsen/neural-networks-and-deep-learning."""

import gzip
import pickle
import numpy as np

def vectorized_label(label):
	"""Turn an MNIST label into a one-hot encoding. For example:
	2 would become the vector: [0,0,1,0,0,0,0,0,0,0], the one is in the
	third element because the encoding is for digits from 0 to 9.""" 
	vectorized_label = np.zeros(10)
	vectorized_label[label] = 1
	return vectorized_label

def load_mnist():
	"""Load the MNIST dataset: mnist.pkl.gz, consisting of 50,000 training 
	images of hand-drawn digits, 10,000 validation images, and 10,000 test 
	images. This file must be stored in the same location as mnist.pkl.gz 
	for this function to work. Note that training_data, validation_data, 
	and test_data are all lists of lists. Each sublist contains 784 
	elements, which represents a flattened version of the corresponding 
	28x28 MNIST image. Both training_labels and validation_labels have 
	been vectorized to make it more convenient for use later, however, 
	test_labels has not been vectorized because the output of the network
	will be compared with the actual number the image represents, not a 
	vectorized version of the number."""
	
	f = gzip.open('mnist.pkl.gz', 'rb')

	training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
	f.close()
	
	# The numbers fed into vectorized_label() are numpy integers, so
	# they first must be cast into integers in order to be used in the 
	# function.
	training_labels = [vectorized_label(int(i)) for i in training_data[1]]
	training_data = training_data[0]

	validation_labels = validation_data[1]
	validation_data = validation_data[0]

	# The test labels are integers rather than vectors because 
	# this makes the comparison between the network output and the 
	# test labels more convenient.
	test_labels = test_data[1]
	test_data = test_data[0]
	
	# Return a tuple of the split MNIST dataset.
	return (training_data, training_labels, validation_data, \
		validation_labels, test_data, test_labels)





