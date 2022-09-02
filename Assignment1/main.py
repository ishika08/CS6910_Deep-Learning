import config
import FeedForwardNN
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import numpy as np
import pdb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import wandb
	

def main(args):
	#print(args)

	# Hyperparameters

	############ Layer Sizes ############

	## Option 1 -- getting from command line with each hidden layer of same/different size
	'''
	#layer_sizes = args.layer_sizes
	'''

	# Option 2 -- getting from command line with each hidden layer of same size
	layer_sizes = []
	layer_sizes.append(784)
	n_hlayers = args.n_hlayers
	hlayer_size = args.hlayer_size
	for i in range(n_hlayers):
		layer_sizes.append(hlayer_size)
	layer_sizes.append(10)
	wandb.log({"n_hidden_layers": n_hlayers, "hidden_layer_size": hlayer_size})

	# Option 3 -- setting layer sizes manually
	'''
	layer_sizes = [784, 128, 32, 10]
	'''


	L = len(layer_sizes)
	epochs = args.epochs
	l_rate = args.l_rate
	optimizer = args.optimizer
	activation_func = args.activation
	loss_func = args.loss
	output_activation = args.output_activation
	batch_size = args.batch_size
	initializer = args.initializer

	labels = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
	lab = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

	################################### Load dataset ###################################           
	(X_train, Y_train), (x_test, y_test) = fashion_mnist.load_data()

	# Display images
	'''
	indexes = [0,1,3,5,6,8,16,18,19,23]
	images = [X_train[i] for i in indexes]
	titles = ['Ankle Boot', 'T-Shirt', 'Dress', 'Pullover', 'Sneaker', 'Sandal', 'Trouser', 'Shirt', 'Coat', 'Bag']
	label = 9
	wandb.log({"Fashion-MNIST images": wandb.Image(images[label], caption=titles[label])})
	wandb.log({"Label": label})
	'''
	
	# Change data type to float64
	X_train = X_train.astype('float64')
	Y_train = Y_train.astype('float64')
	x_test = x_test.astype('float64')
	y_test = y_test.astype('float64')

	# Normalize the images - mean-variance
	scaler = StandardScaler()

	X_train = X_train.reshape(len(X_train),784)
	#X_train = (X_train/255).astype('float32')
	X_train = scaler.fit_transform(X_train)
	Y_train = Y_train.reshape(len(Y_train),1)
	Y_train = to_categorical(Y_train)

	x_test = x_test.reshape(len(x_test), 784)
	#x_test = (x_test/255).astype('float32')
	x_test = scaler.fit_transform(x_test)
	y_test = y_test.reshape(len(y_test), 1)
	y_test = to_categorical(y_test)

	# Split the training dataset into train and validation sets
	x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.10, random_state=42)

	# Creating an object of the class FFNN
	network = FeedForwardNN.FFNN(layer_sizes, L, epochs, l_rate, optimizer, batch_size, activation_func, loss_func, output_activation, initializer)

	# Training the network
	network.train(x_train, y_train, x_val, y_val)

	# Testing
	test_acc, test_loss, y_true, y_pred = network.modelPerformance(x_test, y_test)
	print("################################")
	print("Testing Accuracy = " + str(test_acc))
	print("Testing Loss = " + str(test_loss))
	wandb.log({"test_acc": test_acc})
	wandb.log({"Confusion_Matrix": wandb.sklearn.plot_confusion_matrix(y_true, y_pred, lab)})


if __name__ == "__main__":
	args = config.parseArguments()
	main(args)
