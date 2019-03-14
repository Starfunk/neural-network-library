"""This file implements a neural network library 'from scratch', i.e. 
only using numpy to implement the matrix data structures used to construct 
neural networks. Precisely, you can use this library to create 
fully-connected neural networks; this library does not support the creation 
of CNNs or RNNs. 

Credit: Much of the code and functions in this library were inspired by 
Michael Nielsen's own neural network library: 
https://github.com/mnielsen/neural-networks-and-deep-learning, and his
online textbook: 'Neural networks and deep learning'. Without this
splendid resource, this project would not have been possible.

Indeed, much of the code in this library looks similar to Nielsen's 
network.py and network2.py files. However, there is an aspect of this 
library that distinguishes it from Nielsen's: The biases of the neural 
network are initialized in Nielsen's library to be vertical numpy 
vectors, whereas they are initialized to be horizontal numpy vectors in 
this library. This minor difference turns out to change the specifics 
of the matrix multiplication and arithmetic steps involved in the 
gradient descent and backpropagation functions. 

Another important distinction between Nielsen's library and this 
library is that given the network shape: [2,3,2], that is, a network of 
2 input neurons, a second layer of 3 hidden neurons, and finally an 
output layer of 2 neurons, Nielsen's network.py program outputs a 2x3 
matrix; on the other hand, a network initialized with the shape [2,3,2] 
using this library outputs a 1x2 vector. To me, a 1x2 vector makes much
more sense, as what we are interested in are the activations of
the final layer of neurons and it is easy to interpret the two elements
of a 1x2 vector, [a,b], as the outputs of the final layer of neurons.
It is unclear to me how one is to interpret a 2x3 matrix as the output.
Nevertheless, Nielsen's library served as an invaluable resource when
writing this library and besides these distinguinshing factors, our
libraries remain very similar."""

import numpy as np
import random
import json
import sys
import matplotlib
import matplotlib.pyplot as plt


class Network:
	
	"""The Network class holds the functions needed to initialize a
	neural network and train it. 

	Networks are initialized with the shape given by 'netshape'. For
	example, if netshape = [2,3,2], then the input layer of neurons 
	accepts numpy arrays (NPAs) of the form: [a,b], and outputs NPAs of
	of the form: [c,d]. This example network would have one hidden 
	layer of 3 neurons. Each layer of biases in the network
	are contained in an NPA of shape (n,) where n is the number of 
	neurons in that layer. In our example network, the 
	biases would look something like this:
	
	[	
		array([ 1.24740072, -0.69648469,  2.04505759]), 
	
		array([ 0.39117851, -0.86469781])
	]
	
	This was a set of biases generated using this library for this 
	specific network architecture. Note that there are no biases for 
	the first layer of the network as is the standard convention for 
	neural networks. The first subarray represents the biases for the 
	3 neurons in the second layer. The final subarray represents the 
	biases for the two output neurons. 
	
	The weights are initialized in a similar way. Here, the first 
	subarray holds the weights connecting the first layer to the second 
	layer of the neural network. The first subarray has shape (3, 2) 
	which is determined by the 3 neurons in the second layer, and the
	2 neurons in the first layer. Each row in the 3x2 matrix represents 
	the weights between both neurons in the first layer and one of the
	neurons in the second layer:
	
	[	
		array([[-0.8272883 , -1.74170864],
			   [ 0.22531047,  0.76300333],
               [-0.14128084, -2.00334914]]), 
               
        array([[ 1.43465322, -0.69658175, -0.25336335],
			   [ 0.20888024,  0.00778669,  0.15188696]])
	] 
	
	The first element of this subarray, [-0.8272883 , -1.74170864], 
	is a NPA that represents the 
	weights connecting the two neurons in the first layer to the first
	(or 'top') neuron of the second layer. The remaining NPAs can be 
	similarly interpreted. The values for the weights and 
	biases are initialized with values taken from a normal distribution
	with a mean of 0 and a standard deviation of 1.
	
	Customizable parameters in this model include:
	
	- netshape: The shape of the neural network.
	
	- learning_rate: The rate at which the network learns. In other 
	words, this term controls how large of an impact the gradient 
	descent step has on the weights and biases of the network. If this
	term is too large, the network becomes unstable as it constantly 
	overshoots 'good values' for its weights and biases. However, if
	this term is too small, then it will take an extremely long time
	for the network to learn.
	
	- lmbda: Used in the gradient descent step. Lambda (written as 
	lmbda because 'lambda' is already a reserved word in Python) 
	determines the relative importance of minimizing the weights vs. 
	minimizing the cost function with respect to the weights. In other 
	words, this term controls how much of an impact L2 regularization
	has on the network's weights. 
	
	- mini_batch_size: Determines how large the mini batch is. For
	example, a mini batch size of 32 means each mini batch contains
	32 training images. 
	
	- cost_function: This library contains two cost functions: the
	quadratic cost function and the cross entropy cost function. To
	initialize the network with the quadratic cost function set
	cost_function=QuadraticCost, and for the cross entropy cost 
	function set cost_function=CrossEntropyCost."""
	
	def __init__(self, netshape, learning_rate, mini_batch_size, 
		cost_function, small_weights=False):
		
		# Record the number of layers the network has.
		self.netlength = len(netshape)
		
		# Record the number of neurons in each layer.
		self.netshape = netshape
		
		#Initialize the biases of the network. Each layer of biases is
		#represented as a (1,n) array where n represented the number of
		#neurons in that layer. Each of these numpy arrays (NPAs) 
		#are then stored in the list, biases.
		self.biases = [np.random.randn(1, i)[0] for i in netshape[1:]]
		
		# If the small_weights boolean is set to True, then the 
		# initialized weights have a standard deviation of 1/n where 
		# n is the number
		# neurons in the previous layer relative to the weight.
		# Note that i and j specify, the dimensions of each of the 
		#sublists in the network. So np.random.randn(2, 3) creates an 
		# numpy matrix of dimensions 2 x 3 with values taken from a 
		# normal distribution of mean 0 and standard deviation of 1.
		if small_weights:
			self.weights = [np.random.randn(j, i)/np.sqrt(i) for i, j in 
				zip(netshape[0:], netshape[1:])]
		else:	
			self.weights = [np.random.randn(j, i) for i, j in 
				zip(netshape[0:], netshape[1:])]
		self.learning_rate = learning_rate
		
		# Since the weight decay factor is (eta * lmbda / n), where n 
		# is the size of the dataset (for MNIST, n=50K). So we don't 
		# want to make lambda too small, 5 seems like a reaosnable number
		# and is what Nielsen himself uses in the textbook. While this
		# value is probably not optimal, it is a reasonable value to 
		# start with. 
		self.lmbda = 5
		self.mini_batch_size = mini_batch_size
		self.cost_function = cost_function
		
		
	def feedforward(self, a):
		"""Return the output of the network where 'a' is the input 
		signal. If the softmax boolean value is set to true, then the 
		final layer of neurons will be run through the softmax activation
		function rather than the sigmoid activation function."""		
			
		for b, w in zip(self.biases, self.weights):
				a = np.asarray(a)
				z = np.dot(w, a) + b
				a = sigmoid(z)	
		return a
		
		
	def get_activations(self, a):
		""" Calculates the activations and z values for each layer of 
		neurons in the network. 

		This function is similar to feedforward(), but
		get_activations() was specifically made as a helper function
		for backpropagation() and so it returns two lists: first
		a list containing all the network's activations and a list
		containing every layer's z values. """
			
		activations = [np.asarray(a)]
		zs = []
		for b, w in zip(self.biases, self.weights):
				a = np.asarray(a)
				z = np.dot(w, a) + b
				a = sigmoid(z)
				zs.append(z)
				activations.append(a)
					
		return activations, zs
		
	
	def backpropagation(self, a, y, act_mask, zs_mask): 
		"""Calculate the cost gradient with respect to the weights and 
	biases of the network using the backpropagation algorithm. 'a' is
	the input signal (i.e. in the case of MNIST this would be a list of 
	784 elements. 'y' is the label of the input signal, in the case
	of MNIST, this would be a one-hot encoding of a digit between 0 and 
	9 (e.g. the number 5 would be encoded as [0,0,0,0,0,1,0,0,0,0]."""
		
		# Initialize the gradients of the weights and biases
		# as the same shape as the weights and biases themselves.	
		nabla_b = [np.zeros(b.shape) for b in self.biases] 
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		
		# Propagate the input 'a' through the network to get the
		# activations and z-values in each layer of the network.
		activations, zs = self.get_activations(a)	
		
		# If at last one of the masks is not set to False, then we 
		# run Dropout since this implies Dropout was set to True.
		if act_mask:
			zs = [z * m for z, m in zip(zs, zs_mask)]
			activations = [act * m for act, m in zip(activations, act_mask)]
		
		# Calculate the error (which is returned as a vector) by
		# comparing the network's output with the label 'y'. You can
		# specify which cost function derivative to use when you
		# initialize the network.
		helper = self.cost_function.cost_derivative(activations[-1], y, zs[-1])

		# Note that the error is a 1-D vector of the form 
		# [0, 1, 2, ... , n] where n is the number of output neurons. 
		nabla_b[-1] = helper
		
		# The gradient of the cost function with respect to the 
		# final layer of weights is given by multiplying the transpose of
		# the error vector and the second last layer of activations. This
		# calculation returns an IxJ matrix where I is the length of
		# the error vector and J is the number of neurons in the second
		# layer of the network. This is indeed what we want as the number
		# of rows should be equal to the number of output neurons and
		# the number of columns should correspond with the number of
		# neurons in the second layer.  
		nabla_w[-1] = np.dot(np.asarray([helper]).transpose(), \
			np.asarray([activations[-2]]))
		
		# We have already computed the weights between the last
		# and second last layers (and the biases for the last
		# layer) so our loop begins at the second last layer. The 
		# loop has been reversed because we are working our way
		# backwards through the network.
		for i in reversed(range(self.netlength - 2)):
			# We transpose the weight matrix we just computed and dot it
			# with the previous error vector. Note that the weight matrix
			# is of the form IxJ, where I is the length of
			# the current error vector and J is the number of neurons in the
			# ith layer. Dotting together these terms we get a vector 
			# of length J which represents the number of neurons in the 
			# ith layer.
			helper = np.dot(self.weights[i + 1].transpose(),  helper) * \
				sigmoid_prime(zs[i])
			
			# The gradient for the biases of the ith layer are set 
			# directly to the current error vector. 
			nabla_b[i] = helper
			
			# Exactly as we did above for the weights connecting the 
			# final layer to the second last layer, we first transpose 
			# the error vector (which contains J neurons, the number of 
			# neurons in the ith layer) and dot it with the 
			nabla_w[i] = np.dot(np.asarray([helper]).transpose(), \
				np.asarray([activations[i]]))
		return nabla_b, nabla_w
		
		
	def mini_batch_update_net(self, mini_batch, dataset_length, L2=False,
		dropout=False, dropconnect=False, p=False):
		"""Computes one round of mini-batch gradient descent
		on the network. L2 is a boolean value that determines whether L2
		regularization is used in the gradient descent step. L2 greatly
		improves the network's ability to generalize the features it 
		learns and so it is recommended to have this feature turned on. """

		nabla_b = [np.zeros(b.shape) for b in self.biases] 
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		
		if dropconnect:
			w_mask = [np.random.binomial(1, p, size=w.shape) for w in self.weights]
			# Create the inverse of the w mask
			r_w_mask = [1 - i for i in w_mask]
			# Store the dropped weights
			dropped_weights = [w * m for w, m in zip(self.weights, r_w_mask)]
			#Apply the mask to the weights
			self.weights = [w * m for w, m in zip(self.weights, w_mask)]
		
		# Initialize the activation and z-value masks
		act_mask = False
		zs_mask = False
		if dropout:
			# We get a list of the activations and z-values so that
			# we know how long these lists are, we don't actually
			# use the activation values themselves.
			activations , zs = self.get_activations(mini_batch[0][0])
			# We don't drop out activations in the input or output layers
			# so these masks are simply vectors filled with 1s.
			input_mask = np.ones(len(mini_batch[0][0]))
			output_mask = np.ones(len(activations[-1]))
			# Create the z-value mask for the hidden layers by drawing
			# from a Bernoulli distribution of random variables with
			# probability p of being 1 and (1 - p) of being 0.
			zs_mask = [np.random.binomial(1, p, size=len(i)) for i in zs]
			# Set the final layer of the zs_mask equal to the output
			# mask, this makes sure the output z-values are not dropped.
			zs_mask[-1] = output_mask 
			# Scale zs-mask by p
			zs_mask = [z / p for z in zs_mask]
			# The activation mask is copied from the z-value mask.
			act_mask = zs_mask.copy()
			# The difference is that the activation mask has the input
			# mask as its first layer. 
			act_mask.insert(0, input_mask)
			# Scale activation mask by p
			act_mask = [act / p for act in act_mask]
			
		for a, y in mini_batch:
			# Calculate the gradient for the biases and weights
			# for one training example. 
			d_nabla_b, d_nabla_w = self.backpropagation(a, y, act_mask, zs_mask)
	
			# Calculate the gradient across the mini-batch by summing
			# the individual gradients of each of the training examples.
			# The gradient across the mini-batch is an approximation
			# of the true gradient across the entire training dataset.
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, d_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, d_nabla_w)]
		
		c1 = np.asarray(self.learning_rate / len(mini_batch))
		self.biases = self.biases - c1 * nabla_b
			
		# Calculate the gradient descent step for the weights
		# with L2 regulatization.
		
		if L2:
			c2 = np.asarray(1 - self.learning_rate * self.lmbda / dataset_length)
			self.weights = [c2 * w - c1 * nw for w, nw in zip(self.weights, nabla_w)]
		
		# Calculate the gradient descent step for the weights without 
		# L2 regulatization.
		else:
			self.weights = [w - c1 * nw for w, nw in zip(self.weights, nabla_w)]
		
		# Add back the weights dropped using DropConnect.
		if dropconnect:
			self.weights = [w + r for w, r in zip(self.weights, dropped_weights)]
		
		
	def stochastic_gradient_descent(self, dataset, labels, epochs, L2=False, 
		monitor=False, test_data=False, test_labels=False, dropout=False,
		dropconnect=False, p=False):
		"""Perform mini batch stochastic gradient descent on the neural 
		network using the dataset and labels provided by the function 
		arguments. L2 is a boolean value that determines whether L2
		regularization is used in the gradient descent step. L2 greatly
		improves the network's ability to generalize the features it 
		learns and so it is recommended to have this feature turned on.
		The variable, monitor, determines if 
		
		the program returns explicit information about the performance of 
		the neural network. In order to activate 'monitor', assign to it
		the list: [test_data, test_labels]. Then after each training epoch
		---i.e. after each mini-batch gradient descent step---the function
		will print the network's current peformance on the test data. 
		Note that if monitor is set to True, you must input the test data
		and test labels you wish to compare for the network. 
		
		After the final training epoch, the function will plot the performance
		of the model (network and model are used interchangably here) across
		all training epochs on the test data. Note that having monitor on
		significantly slows down the training process as the program has to
		run many more feedforward signals through the network after every
		training epoch.""" 
	
		if monitor and not (test_data.any and test_labels.any):
			print("You must input the test data and test labels!")
			exit()
		
		if (dropout or dropconnect) and not p:
			print("You must input the probability, p, if you are \
			using dropout or dropconnect!")
			exit()
			
		dataset_length = len(dataset)
	
		if monitor:
			epochs_list = [i for i in range(epochs)]
			evaluations = []
				
			for i in range(epochs):
				samples = np.random.randint(dataset_length, size=self.mini_batch_size)
				mini_batch = []
				for j in samples:
					training_example = dataset[j]
					label = labels[j]
					mini_batch.append([training_example, label])
					
				self.mini_batch_update_net(mini_batch, dataset_length, L2=L2)	
				
				accuracy = self.evaluation(test_data, test_labels)
				evaluations.append(accuracy)
				print("Epoch: " + str(i) + 
					", The NN correctly classified: " + ("{0:.2f}".format(accuracy)) 
					+ "% of the test dataset")
					
			plt.plot(epochs_list, evaluations)
			plt.xlabel('Epoch')
			plt.ylabel('Accuracy on test data')
			plt.title('Evaluating model performance')
			plt.show()
		
		else:
			for i in range(epochs):
				print("Epoch: " + str(i))
				samples = np.random.randint(dataset_length, size=self.mini_batch_size)
				mini_batch = []
				
				for j in samples:
					training_example = dataset[j]
					label = labels[j]
					mini_batch.append([training_example, label])
					
				self.mini_batch_update_net(mini_batch, dataset_length, L2=L2)	
		

	def evaluation(self, dataset, labels):
		"""Compare the output of the neural network with the set of labels. 
		This function was specifically made to test network performance on 
		the MNIST dataset. Here, we interpret the position of the highest
		value in the network's output list to be the network's 'guess' of
		which number the input list represents. The output of this function
		is the percentage of correct 'guesses' the network made on the 
		dataset."""	
		
		error = 0
		l = len(dataset)	

		for data, label in zip(dataset, labels):

			if np.argmax(self.feedforward(data)) == label:
					continue
			else:
				error += 1		
		accuracy = (l - error)/ l * 100	
		return accuracy	
		
		
	def cost_evaluation(self, dataset, labels): 
		"""Returns the cost the network produces on the input dataset.
		This is an alternative method for evaluating the performance of
		the model by seeing how well it minimzes the cost over several
		training epochs (and this functions role is to return the cost
		for one training epoch)."""
		
		dataset_length = len(dataset)
		cost = 0
		
		for data, label in zip(dataset, labels):
			# The cost function returns the cost as a numpy array, so
			# we sum up the elements and that becomes the 'magnitude'
			# of the error. 
			cost += np.sum(self.cost_function.cost(data, label))
		
		# The cost is the average over the dataset, therefore, we 
		# divide the sum by the size of the dataset.
		cost = (1 / dataset_length) * cost 
		return cost

	
	def save(self, filename=False):
		"""Save the network in a JSON format. Note that numpy arrays cannot 
		be saved in the JSON format, so we must first convert the biases
		and weights to be lists before saving the network as a JSON. The
		JSON is saved in a text file in the same directory as this file. By
		default, the saved file is called 'network.txt' but this can be 
		changed if another file name is given in the function call."""
	
		biases_list = [i.tolist() for i in self.biases]
		weights_list = [i.tolist() for i in self.weights]
		data = {
			"netlength": self.netlength,
			"netshape": self.netshape,
			"biases": biases_list,
			"weights": weights_list,
			"learning_rate": self.learning_rate,
			"mini_batch_size": self.mini_batch_size,
			"cost_function": str(self.cost_function.__name__),
		}
		
		if filename:
			with open(filename, 'w') as outfile:  
				json.dump(data, outfile)
		else:
			with open('network.txt', 'w') as outfile:  
				json.dump(data, outfile)
			

def load(filename):	
	"""Load a saved neural network. This function initializes a network 
		instance with the parameters of the loaded network. Note that the 
	file with the network parameters should be in the same directory
	as this file."""
	
	with open(filename) as json_file:  
		network_data = json.load(json_file)
		
	net = Network([network_data['netshape']], network_data['learning_rate'],
		network_data['mini_batch_size'], \
		cost_function=getattr(sys.modules[__name__], network_data['cost_function']))
	
    # We need to reconvert the sublists in the biases and weights lists
    # back to being numpy arrays.
	biases_numpy = [np.asarray(i) for i in network_data['biases']]
	weights_numpy = [np.asarray(i) for i in network_data['weights']]

	net.biases = biases_numpy
	net.weights = weights_numpy
	net.netlength = network_data['netlength']
	return net


def compare_net(net1, net2, training_data, training_labels, test_data,
	test_labels, epochs, mini_batch_size, label1, label2, L2_test=False,
	L2=False, dropout_test=False, dropconnect_test=False, p=False):
	"""Compares the performance of two neural networks and plots the 
	accuracy values of both networks using matplotlib. Mini-batch 
	stochastic gradient descent is used to train both networks for
	a number of epochs (specified by the variable "epochs"."""
	
	epochs_list = [i for i in range(epochs)]
	
	evaluations1 = []
	evaluations2 = []
	dataset_length = len(training_data)
	
	if dropout_test:
		net1_dropout = True
	else: 
		net1_dropout = False
		
	if dropconnect_test:
		net1_dropconnect = True
	else: 
		net1_dropconnect = False
	
	# If the L2 boolean is set to True, then both networks use L2
	# regularization. 
	if L2:
		net1_L2 = True
		net2_L2 = True
		
	# If the L2 test boolean is set to True, then we are testing the
	# effect of L2 regularization and so net1 uses L2 regularization,
	# while net2 does not.
	elif L2_test:
		net1_L2 = True
		net2_L2 = False
	
	# Otherwise, we are not using L2 regularization in our networks.
	else:
		net1_L2 = False
		net2_L2 = False
		
	for i in range(epochs):
		samples = np.random.randint(dataset_length, size=mini_batch_size)
		mini_batch = []
		for j in samples:
			training_example = training_data[j]
			label = training_labels[j]
			mini_batch.append([training_example, label])
						
		net1.mini_batch_update_net(mini_batch, dataset_length, L2=net1_L2,
		 dropout=net1_dropout, dropconnect=net1_dropconnect, p=p)
		 	
		net2.mini_batch_update_net(mini_batch, dataset_length, L2=net2_L2)
					
		acc1 = net1.evaluation(test_data, test_labels)
		acc2 = net2.evaluation(test_data, test_labels)
		
		evaluations1.append(acc1)
		evaluations2.append(acc2)
		
		print("Epoch: " + str(i) + ", net1 correctly classified: " 
			+ ("{0:.2f}".format(acc1))) 
		print("Epoch: " + str(i) + ", net2 correctly classified: " 
			+ ("{0:.2f}".format(acc2)))
		print()
						
	fig, ax = plt.subplots()

	ax.plot(epochs_list, evaluations1, label=label1)
	ax.plot(epochs_list, evaluations2, label=label2)
	
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy on test data')
	plt.title('Comparing model performance')				
	plt.legend()
	plt.show()
						
	
def sigmoid(z):
	"""The sigmoid activation function. The sigmoid function 
	transforms all values on the number line to values between 0 and 1."""
	return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
	"""The derivative of the sigmoid activation function."""
	return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
	"""The softmax activation function. Can be used as the final
	activation function for the network."""	
	
	total = np.exp(z).sum()
	z_exp = np.exp(z)
	z[True] = total #Convert every element to be equal to total
	return z_exp / total	
	
def tanh(z):
	"""The tanh activation function. The tanh function transforms all values
	on the number line to values between -1 and 1."""
	e = np.exp(2 * z)
	return ((e - 1) / (e + 1))


def tanh_prime(z):
	"""The derivative of the tanh function."""
	e = np.exp(2 * z)
	return (4 * e) / (e + 1) ** 2


def relu(z):
	"""The ReLU activation function. ReLu is defined as f(x) = max(0,x)."""
	
	flag = isinstance(z, np.ndarray) # Test if z is an array
	if flag:
		z[z < 0] = 0
		return z
	else:
		z = max(0,z)
		return z
	return z
	

def relu_prime(z):
	"""The derivative of the ReLU activation function. Similar to relu(),
	first test if z is a list or integer and then transform the values 
	accordingly."""
	
	flag = isinstance(z, np.ndarray)
	if flag:
		z[z < 0] = 0
		z[z >= 0] = 1
		return z
	else:
		if z < 0:
			return 0
		else: 
			return 1 	

			
def linear(z):
	"""The linear activation function"""
	return z
	

def linear_prime(z):
	"""The derivative of the linear activation function"""
	return 1

class QuadraticCost: 
	
	def cost(a, y):
		"""Calculate the cost using the quadratic cost function. This
		function is used in the cost_evalaution function to assess
		how good the model is at minimizing the cost on the given 
		dataset."""
		return 0.5 * (a - y) ** 2
	
	def cost_derivative(a, y, z):
		"""The derivative of the quadratic cost function. This function is used 
		in backpropagation to calculate the error between the output of the 
		network and the label. Where 'a' is the network input, 'y' is the
		label, and """
		return (a - y) * sigmoid_prime(z)
	
class CrossEntropyCost:
	
	def cost(a, y):
		"""Calculate the cost using the cross entropy cost function. This
		function is used in the cost_evalaution function to assess
		how good the model is at minimizing the cost on the given 
		dataset."""
		
		# Python can return a variety of warnings when using np.log(). 
		# So first we suppress these warnings. We use the np.nan_to_num  
		# function, to convert nan values to 0. For example, if y and
		# and a both equal 1, then (1 - y) * np.log(1 - a) would
		# normally output nan, however, using this function, the
		# output becomes 0, which is indeed what we want. 
		with np.errstate(divide='ignore', invalid='ignore'):
			return np.nan_to_num(-y * np.log(a)) - np.nan_to_num((1 - y) * np.log(1 - a))

	def cost_derivative(a, y, z):
		"""The derivative of the cross-entropy cost function. This function is 
		used in backpropagation to calculate the error between the output of the 
		network and the label. Where 'a' is the network input and 'y' is the
		label. We include the input argument of z to be consistent
		with the other cost functions."""
		return (a - y)
