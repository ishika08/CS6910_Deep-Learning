import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pdb
import math
from tqdm import tqdm
import wandb
np.random.seed(1)

wandb.init(config={"batch_size": 32, "l_rate": 0.001, "optimizer": 'nadam', "epochs": 5, "activation": "relu", "initializer": "random", "loss": "squared_error", "n_hlayers": 5, "hlayer_size": 128}, project="Deep-Learning")
myconfig = wandb.config

class FFNN():
	# Initializing the hyperparameters
	def __init__(self, layer_sizes, L, epochs=10, l_rate=0.001, optimizer='sgd', batch_size=16, activation_func='sigmoid', loss_func='cross_entropy', output_activation_func='softmax', initializer='xavier'):
		
		self.layer_sizes = layer_sizes		# Size of each layer			
		self.L = L				# Number of layer

		'''
		self.epochs = myconfig.epochs			# Total number of epochs
		self.l_rate = myconfig.l_rate			# Learning rate
		self.optimizer = myconfig.optimizer		# Optimization algorithm
		self.batch_size = myconfig.batch_size		# Size of a batch
		self.activation_func = myconfig.activation	# Activation funtion for the hidden layers
		self.initializer = myconfig.initializer		# For initializing wights
		self.loss_func = myconfig.loss			# Loss funtion

		'''
		self.epochs = epochs			# Total number of epochs
		self.l_rate = l_rate			# Learning rate
		self.optimizer = optimizer		# Optimization algorithm
		self.batch_size = batch_size		# Size of a batch
		self.activation_func = activation_func	# Activation funtion for the hidden layers
		self.initializer = initializer		# For initializing wights
		self.loss_func = loss_func			# Loss funtion
		
		
		self.output_activation_func = output_activation_func	# Activation funtion for the output layer
		self.parameters = self.initializeModelParameters()	# Initializing the parameters -- weights and biases

		print()
		print("############## Hyperparameters Values ##############")
		print("Number of Hidden Layers: " + str(self.L - 2))
		print("Layer Sizes: " + str(self.layer_sizes))
		print("Number of Epochs: " + str(self.epochs))
		print("Learning Rate: " + str(self.l_rate))
		print("Optimizer: " + self.optimizer)
		print("Batch Size: " + str(self.batch_size))
		print("Activation Function: " + self.activation_func)
		print("Loss Function: " + self.loss_func)
		print("Output Activation Funtion: " + self.output_activation_func)
		print("Initializer: " + self.initializer)
		print("###################################################")
		print()

	# Activation funtion for the hidden layers
	def activation(self, x, derivative=False):
		if self.activation_func == 'sigmoid':
			if derivative:
				return (np.exp(-x))/((np.exp(-x)+1)**2)
			return 1/(1 + np.exp(-x))
		elif self.activation_func == 'tanh':
			gz = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
			if derivative:
				return 1 - (gz)**2
			return gz
		elif self.activation_func == 'relu':
			if derivative:
				return 1. * (x > 0)
			return x * (x > 0)

	# Activation funtion for the output layer
	def outputActivation(self, x, derivative=False):
		if self.output_activation_func == 'softmax':
			exps = np.exp(x - x.max())
			if derivative:
			    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
			return exps / np.sum(exps, axis=0)

	# Initializing the parameters -- weights and biases
	def initializeModelParameters(self):
		parameters = {}
		for l in range(1, self.L):
			if self.initializer == 'random':
				parameters["W" + str(l)] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l - 1])*0.1
			elif self.initializer == 'xavier':
				parameters["W" + str(l)] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l - 1]) * np.sqrt(2/ (self.layer_sizes[l - 1] + self.layer_sizes[l]))
			parameters["b" + str(l)] = np.zeros((self.layer_sizes[l], 1))
		return parameters

	# Initialize gradients
	def initialize_gradients(self):
		gradients = {}
		for l in range(1, self.L):
			gradients["W" + str(l)] = np.zeros((self.layer_sizes[l], self.layer_sizes[l - 1]))
			gradients["b" + str(l)] = np.zeros((self.layer_sizes[l], 1))
		return gradients
		

	# Forward Propogation
	def forwardPropagation(self, x):
		pre_activations = {}
		activations = {}

		activations['h0'] = x.reshape(len(x),1)

		# From layer 1 to L-1
		for i in range(1, self.L-1):
			pre_activations['a' + str(i)] = self.parameters['b' + str(i)] + np.matmul(self.parameters['W' + str(i)], activations['h' + str(i-1)])
			activations['h' + str(i)] = self.activation(pre_activations['a' + str(i)], derivative=False)

		# Last layer L
		pre_activations['a' + str(self.L-1)] = self.parameters['b' + str(self.L-1)] + + np.matmul(self.parameters['W' + str(self.L-1)], activations['h' + str(self.L-1-1)])
		activations['h' + str(self.L-1)] = self.outputActivation(pre_activations['a' + str(self.L-1)])

		return activations, pre_activations
		

	# Back Propogation
	def backwardPropagation(self, y, activations, pre_activations):
		gradients = {}

		# Compute output gradient
		f_x = activations['h' + str(self.L-1)]
		e_y = y.reshape(len(y), 1)

		# Gardient with respect to last layer
		if self.loss_func == 'cross_entropy':
			gradients['a' + str(self.L-1)] = (f_x - e_y)
		elif self.loss_func == 'squared_error':
			gradients['a' + str(self.L-1)] = (f_x - e_y)*f_x*(1-f_x)

		# Compute gradients for hidden layers
		for k in range(self.L-1, 0, -1):
			# Compute gradients with respect to paramters
			gradients['W' + str(k)] = np.outer(gradients['a' + str(k)], activations['h' + str(k-1)])
			gradients['b' + str(k)] = gradients['a' + str(k)]

			# Compute gradients with respect to layer below
			gradients['h' + str(k-1)] = np.dot(self.parameters['W' + str(k)].T, gradients['a' + str(k)])
	
			# Compute gradients with respect to layer below (pre-activation)
			if k > 1:
				gradients['a' + str(k-1)] = gradients['h' + str(k-1)] * self.activation(pre_activations['a' + str(k-1)], derivative=True)	

		return gradients

	# Compute Loss
	def computeLoss(self, yHat, y):
		if self.loss_func == 'cross_entropy':
			indexClass = np.argmax(y)
			prob = yHat[indexClass]
			if(prob<=0):
				prob = prob + 0.0000000001		
			loss = -math.log(prob)
			return loss
		if self.loss_func == 'squared_error':
			loss = 0.5*np.sum((y-yHat)**2)
			return loss
		
	# Find the accuracy
	def modelPerformance(self, x_test, y_test):
		predictions = []
		y_true = []
		y_pred = []
		losses = []
		for x,y in tqdm(zip(x_test ,y_test), total=len(x_test)):
			activations, pre_activations = self.forwardPropagation(x)
			predictedClass = np.argmax(activations['h' + str(self.L-1)])
			y.reshape(len(y),1)
			actualClass = np.argmax(y)
			y_true.append(actualClass)
			y_pred.append(predictedClass)
			predictions.append(predictedClass == actualClass)
			losses.append(self.computeLoss(activations['h' + str(self.L-1)],y))

		accuracy = (np.sum(predictions)*100)/len(predictions)
		loss = np.sum(losses)/len(losses)
			
		return accuracy, loss, y_true, y_pred


	# Optimization Algorithm: Stochastic Gradient Descent
	def do_stochastic_gradient_descent(self, x_train, y_train, x_val, y_val):

		# Learning rate
		eta = self.l_rate

		for epoch in range(self.epochs):
			print(" =============== Epoch Number: " + str(epoch+1) + " =============== ")

			# Initialize the gradients
			grads = self.initialize_gradients()

			# Learn the parameters
			for x,y in tqdm(zip(x_train,y_train), total=len(x_train)):

				# Forward Propagation
				activations, pre_activations = self.forwardPropagation(x)

				# Backward Propagation
				current_gradients = self.backwardPropagation(y, activations, pre_activations)
	
				# Update paramters
				for key in self.parameters:
					self.parameters[key] = self.parameters[key] - eta*current_gradients[key]
		
			# Validation Accuracy
			train_acc, train_loss, y_true, y_pred = self.modelPerformance(x_train, y_train)
			val_acc, val_loss, y_true, y_pred = self.modelPerformance(x_val, y_val)
			print("Training Accuracy = " + str(train_acc))
			print("Training Loss = " + str(train_loss))
			print("Validation Accuracy = " + str(val_acc))
			print("Validation Loss = " + str(val_loss))
			wandb.log({"val_acc": val_acc, "train_acc": train_acc, "val_loss": val_loss, "train_loss": train_loss, "epoch": epoch+1})


	# Optimization Algorithm: Moment Based Gradient Descent
	def do_moment_based_gradient_descent(self, x_train, y_train, x_val, y_val):
		
		# Learning rate
		eta = self.l_rate

		# Gamma value
		gamma = 0.9

		# Previous values -- History
		prev_gradients = self.initialize_gradients()

		for epoch in range(self.epochs):
			print(" =============== Epoch Number: " + str(epoch+1) + " =============== ")

			# Initialize the gradients
			grads = self.initialize_gradients()
			lookAheads = self.initialize_gradients()

			# Number of points seen
			num_points_seen = 0

			# Learn the parameters
			for x,y in tqdm(zip(x_train,y_train), total=len(x_train)):

				# Forward Propagation
				activations, pre_activations = self.forwardPropagation(x)

				# Backward Propagation
				current_gradients = self.backwardPropagation(y, activations, pre_activations)

				# Accumulate gradients
				for key in grads:
					grads[key] = grads[key] + current_gradients[key]

				num_points_seen = num_points_seen + 1

				if(num_points_seen % self.batch_size == 0):

					# Calculate Look Ahead
					for key in lookAheads:
						lookAheads[key] = gamma*prev_gradients[key] + eta*grads[key]

					# Update Parameters
					for key in self.parameters:
						self.parameters[key] = self.parameters[key] - lookAheads[key]

					# Update History
					for key in prev_gradients:
						prev_gradients[key] = lookAheads[key]

					# Initialize the gradients
					grads = self.initialize_gradients()
		
			# Validation Accuracy
			train_acc, train_loss, y_true, y_pred = self.modelPerformance(x_train, y_train)
			val_acc, val_loss, y_true, y_pred = self.modelPerformance(x_val, y_val)
			print("Training Accuracy = " + str(train_acc))
			print("Training Loss = " + str(train_loss))
			print("Validation Accuracy = " + str(val_acc))
			print("Validation Loss = " + str(val_loss))
			wandb.log({"val_acc": val_acc, "train_acc": train_acc, "val_loss": val_loss, "train_loss": train_loss, "epoch": epoch+1})
		

	# Optimization Algorithm: Nesterov Accelerated Gradient Descent
	def do_nesterov_accelerated_gradient_descent(self, x_train, y_train, x_val, y_val):
		
		# Learning rate
		eta = self.l_rate

		# Gamma value
		gamma = 0.95

		# Previous values -- History
		prev_gradients = self.initialize_gradients()

		for epoch in range(self.epochs):
			print(" =============== Epoch Number: " + str(epoch+1) + " =============== ")

			# Initialize the gradients
			grads = self.initialize_gradients()			# For accumulating gradients
			lookAheads = self.initialize_gradients()		# Lookaheads

			# Calculate Look Ahead
			for key in lookAheads:
				lookAheads[key] = gamma*prev_gradients[key]

			# Update parameters based on lookahead
			for key in self.parameters:
				self.parameters[key] = self.parameters[key] - lookAheads[key]

			# Number of points seen
			num_points_seen = 0
			
			# Learn the parameters
			for x,y in tqdm(zip(x_train,y_train), total=len(x_train)):

				# Forward Propagation
				activations, pre_activations = self.forwardPropagation(x)

				# Backward Propagation
				current_gradients = self.backwardPropagation(y, activations, pre_activations)

				# Accumulate gradients
				for key in grads:
					grads[key] = grads[key] + current_gradients[key]
				
				num_points_seen = num_points_seen + 1

				if(num_points_seen % self.batch_size == 0):	

					# Calculate Look Ahead
					for key in lookAheads:
						lookAheads[key] = gamma*prev_gradients[key] + eta*grads[key]

					# Update Parameters
					for key in self.parameters:
						self.parameters[key] = self.parameters[key] - lookAheads[key]

					# Update History
					for key in prev_gradients:
						prev_gradients[key] = lookAheads[key]

					# Initialize the gradients
					grads = self.initialize_gradients()
		
			# Validation Accuracy
			train_acc, train_loss, y_true, y_pred = self.modelPerformance(x_train, y_train)
			val_acc, val_loss, y_true, y_pred = self.modelPerformance(x_val, y_val)
			print("Training Accuracy = " + str(train_acc))
			print("Training Loss = " + str(train_loss))
			print("Validation Accuracy = " + str(val_acc))
			print("Validation Loss = " + str(val_loss))
			wandb.log({"val_acc": val_acc, "train_acc": train_acc, "val_loss": val_loss, "train_loss": train_loss, "epoch": epoch+1})


	# Optimization Algorithm: RMSProp
	def do_rmsprop(self, x_train, y_train, x_val, y_val):
		
		# Learning rate
		eta = self.l_rate

		# Beta value
		beta = 0.9

		# Epsilon
		eps = 0.00000001

		# Previous values -- History
		lookAheads = self.initialize_gradients()

		for epoch in range(self.epochs):
			print(" =============== Epoch Number: " + str(epoch+1) + " =============== ")

			# Initialize the gradients
			grads = self.initialize_gradients()

			# Number of points seen
			num_points_seen = 0

			# Learn the parameters
			for x,y in tqdm(zip(x_train,y_train), total=len(x_train)):

				# Forward Propagation
				activations, pre_activations = self.forwardPropagation(x)

				# Backward Propagation
				current_gradients = self.backwardPropagation(y, activations, pre_activations)

				# Accumulate gradients
				for key in grads:
					grads[key] = grads[key] + current_gradients[key]

				num_points_seen = num_points_seen + 1

				if(num_points_seen % self.batch_size == 0):

					# Update History
					for key in lookAheads:
						lookAheads[key] = beta*lookAheads[key] + (1-beta)*np.square(grads[key])

					# Update Parameters
					for key in self.parameters:
						self.parameters[key] = self.parameters[key] - (eta/np.sqrt(lookAheads[key] + eps))*grads[key]

					# Initialize the gradients
					grads = self.initialize_gradients()
		
			# Validation Accuracy
			train_acc, train_loss, y_true, y_pred = self.modelPerformance(x_train, y_train)
			val_acc, val_loss, y_true, y_pred = self.modelPerformance(x_val, y_val)
			print("Training Accuracy = " + str(train_acc))
			print("Training Loss = " + str(train_loss))
			print("Validation Accuracy = " + str(val_acc))
			print("Validation Loss = " + str(val_loss))
			wandb.log({"val_acc": val_acc, "train_acc": train_acc, "val_loss": val_loss, "train_loss": train_loss, "epoch": epoch+1})


	# Optimization Algorithm: Adam
	def do_adam(self, x_train, y_train, x_val, y_val):
		
		first_momenta = self.initialize_gradients()
		second_momenta = self.initialize_gradients()
		first_momenta_hat = self.initialize_gradients()
		second_momenta_hat = self.initialize_gradients()

		# Learning rate
		eta = self.l_rate

		# Beta value
		beta1 = 0.9
		beta2 = 0.999

		# Epsilon
		eps = 0.00000001

		for epoch in range(self.epochs):
			print(" =============== Epoch Number: " + str(epoch+1) + " =============== ")

			# Initialize the gradients
			grads = self.initialize_gradients()

			# Number of points seen
			num_points_seen = 0

			# Learn the parameters
			for x,y in tqdm(zip(x_train,y_train), total=len(x_train)):

				# Forward Propagation
				activations, pre_activations = self.forwardPropagation(x)

				# Backward Propagation
				current_gradients = self.backwardPropagation(y, activations, pre_activations)

				# Accumulate gradients
				for key in grads:
					grads[key] = grads[key] + current_gradients[key]

				num_points_seen = num_points_seen + 1

				if(num_points_seen % self.batch_size == 0):

					# Update History
					for key in self.parameters:
						first_momenta[key] = beta1*first_momenta[key] + (1-beta1)*grads[key]
						second_momenta[key] = beta2*second_momenta[key] + (1-beta2)*np.square(grads[key])
						first_momenta_hat[key] = first_momenta[key]/(1-math.pow(beta1, epoch+1))
						second_momenta_hat[key] = second_momenta[key]/(1-math.pow(beta2, epoch+1))
						self.parameters[key] = self.parameters[key] - (eta/(np.sqrt(second_momenta_hat[key]) + eps))*first_momenta_hat[key]

					# Initialize the gradients
					grads = self.initialize_gradients()
		
			# Validation Accuracy
			train_acc, train_loss, y_true, y_pred = self.modelPerformance(x_train, y_train)
			val_acc, val_loss, y_true, y_pred = self.modelPerformance(x_val, y_val)
			print("Training Accuracy = " + str(train_acc))
			print("Training Loss = " + str(train_loss))
			print("Validation Accuracy = " + str(val_acc))
			print("Validation Loss = " + str(val_loss))
			wandb.log({"val_acc": val_acc, "train_acc": train_acc, "val_loss": val_loss, "train_loss": train_loss, "epoch": epoch+1})


	# Optimization Algorithm: NAdam
	def do_nadam(self, x_train, y_train, x_val, y_val):
		
		first_momenta = self.initialize_gradients()
		second_momenta = self.initialize_gradients()
		first_momenta_hat = self.initialize_gradients()
		second_momenta_hat = self.initialize_gradients()

		# Learning rate
		eta = self.l_rate

		# Beta value
		beta1 = 0.9
		beta2 = 0.999

		# Epsilon
		eps = 0.00000001

		# Gamma
		gamma = 0.95

		# Previous values -- History
		prev_gradients = self.initialize_gradients()

		for epoch in range(self.epochs):
			print(" =============== Epoch Number: " + str(epoch+1) + " =============== ")

			# Initialize the gradients
			grads = self.initialize_gradients()
			lookAheads = self.initialize_gradients()

			# Number of points seen
			num_points_seen = 0

			# Calculate Look Ahead
			for key in lookAheads:
				lookAheads[key] = gamma*prev_gradients[key]

			# Update parameters based on lookahead
			for key in self.parameters:
				self.parameters[key] = self.parameters[key] - lookAheads[key]

			# Learn the parameters
			for x,y in tqdm(zip(x_train,y_train), total=len(x_train)):

				# Forward Propagation
				activations, pre_activations = self.forwardPropagation(x)

				# Backward Propagation
				current_gradients = self.backwardPropagation(y, activations, pre_activations)

				# Accumulate gradients
				for key in grads:
					grads[key] = grads[key] + current_gradients[key]

				num_points_seen = num_points_seen + 1

				if(num_points_seen % self.batch_size == 0):

					# Calculate Look Ahead
					for key in lookAheads:
						lookAheads[key] = gamma*prev_gradients[key] + eta*grads[key]

					# Update History
					for key in self.parameters:
						first_momenta[key] = beta1*first_momenta[key] + (1-beta1)*grads[key]
						second_momenta[key] = beta2*second_momenta[key] + (1-beta2)*np.square(grads[key])
						first_momenta_hat[key] = first_momenta[key]/(1-math.pow(beta1, epoch+1))
						second_momenta_hat[key] = second_momenta[key]/(1-math.pow(beta2, epoch+1))
						self.parameters[key] = self.parameters[key] - (eta/(np.sqrt(second_momenta_hat[key]) + eps))*first_momenta_hat[key]

					# Update History
					for key in prev_gradients:
						prev_gradients[key] = lookAheads[key]

					# Initialize the gradients
					grads = self.initialize_gradients()
		
			# Validation Accuracy
			train_acc, train_loss, y_true, y_pred = self.modelPerformance(x_train, y_train)
			val_acc, val_loss, y_true, y_pred = self.modelPerformance(x_val, y_val)
			print("Training Accuracy = " + str(train_acc))
			print("Training Loss = " + str(train_loss))
			print("Validation Accuracy = " + str(val_acc))
			print("Validation Loss = " + str(val_loss))
			wandb.log({"val_acc": val_acc, "train_acc": train_acc, "val_loss": val_loss, "train_loss": train_loss, "epoch": epoch+1})


	# Training the model
	def train(self, x_train, y_train, x_val, y_val):
		if self.optimizer == 'sgd':
			self.do_stochastic_gradient_descent(x_train, y_train, x_val, y_val)
		elif self.optimizer == 'mgd':
			self.do_moment_based_gradient_descent(x_train, y_train, x_val, y_val)
		elif self.optimizer == 'nag':
			self.do_nesterov_accelerated_gradient_descent(x_train, y_train, x_val, y_val)
		elif self.optimizer == 'rmsprop':
			self.do_rmsprop(x_train, y_train, x_val, y_val)
		elif self.optimizer == 'adam':
			self.do_adam(x_train, y_train, x_val, y_val)
		elif self.optimizer == 'nadam':
			self.do_nadam(x_train, y_train, x_val, y_val)
		return 0




