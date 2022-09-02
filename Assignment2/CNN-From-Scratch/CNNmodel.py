import numpy as np
import keras
import cv2
import glob
import pdb
import os
from tqdm import tqdm

from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, LeakyReLU, ReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras import backend as K 
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy, binary_accuracy, binary_crossentropy
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping

from numpy.random import seed
from tensorflow.compat.v1 import set_random_seed

seed(1)
set_random_seed(42)

WIDTH, HEIGHT, CHANNELS = 224, 224, 3

# In order to run Wandb
WANDB = 0

if WANDB:
	import wandb
	from wandb.keras import WandbCallback
	wandb.init(config={"batch_size": 64, "l_rate": 0.001, "optimizer": "Adam", "epochs": 10, "activation": "leakyrelu", "denselayer_size": 128, "dropout": 0.5}, project="Deep-Learning-CNN")
	myconfig = wandb.config

# Class names
class_names = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia']


# CNN Model Class
class CNN_Model():

	##################### Initialize the Hyperparameters #####################
	def __init__(self, n_classes, n_filters, filter_size, filter_multiplier, var_n_filters, l_rate, epochs, optimizer, activation, loss, batch_size, initializer, data_augmentation, denselayer_size, batch_norm, train_model, dropout):

		self.n_classes = n_classes
		self.n_filters = n_filters
		self.filter_size = filter_size
		self.filter_multiplier = filter_multiplier
		self.var_n_filters = var_n_filters
		self.l_rate = l_rate
		self.epochs = epochs
		self.optimizer = optimizer
		self.activation = activation
		self.loss = loss
		self.batch_size = batch_size
		self.initializer = initializer
		self.data_augmentation = data_augmentation
		self.denselayer_size = denselayer_size
		self.batch_norm = batch_norm
		self.train_model = train_model
		self.dropout = dropout

		self.n_filters_layer1 = int(n_filters)
		self.n_filters_layer2 = int(self.n_filters_layer1*filter_multiplier)
		self.n_filters_layer3 = int(self.n_filters_layer2*filter_multiplier)
		self.n_filters_layer4 = int(self.n_filters_layer3*filter_multiplier)
		self.n_filters_layer5 = int(self.n_filters_layer4*filter_multiplier)

		self.filter_shape = (self.filter_size, self.filter_size)

		self.model = self.InitializeModel()

	##################### Defining Model Architecture #####################
	def InitializeModel(self):
		
		if K.image_data_format() == 'channels_first':
		    input_shape = (CHANNELS, WIDTH, HEIGHT)
		else:
		    input_shape = (WIDTH, HEIGHT, CHANNELS)


		############ Initialize Model ############
		model = Sequential()

		initializer = keras.initializers.Orthogonal(gain=1.0, seed=42)

		### Convolution
		model.add(Conv2D(self.n_filters_layer1, self.filter_shape, activation='linear', kernel_initializer=initializer, padding='same', input_shape=input_shape))
		### Activation		
		if self.activation == "leakyrelu":		
			model.add(LeakyReLU(alpha=0.1))
		elif self.activation == "relu":
			model.add(Activation('relu'))
		### Batch-Normalization
		if self.batch_norm:
			model.add(BatchNormalization())
		### Max-Pooling
		model.add(MaxPooling2D((2, 2), padding='same'))

		### Convolution
		model.add(Conv2D(self.n_filters_layer2, self.filter_shape, activation='linear', kernel_initializer=initializer, padding='same'))
		### Activation
		if self.activation == "leakyrelu":		
			model.add(LeakyReLU(alpha=0.1))
		elif self.activation == "relu":
			model.add(Activation('relu'))
		### Batch-Normalization
		if self.batch_norm:
			model.add(BatchNormalization())
		### Max-Pooling
		model.add(MaxPooling2D((2, 2), padding='same'))

		### Convolution
		model.add(Conv2D(self.n_filters_layer3, self.filter_shape, activation='linear', kernel_initializer=initializer, padding='same'))
		### Activation
		if self.activation == "leakyrelu":		
			model.add(LeakyReLU(alpha=0.1))
		elif self.activation == "relu":
			model.add(Activation('relu'))
		### Batch-Normalization
		if self.batch_norm:
			model.add(BatchNormalization())
		### Max-Pooling
		model.add(MaxPooling2D((2, 2), padding='same'))

		### Convolution
		model.add(Conv2D(self.n_filters_layer4, self.filter_shape, activation='linear', kernel_initializer=initializer, padding='same'))
		### Activation
		if self.activation == "leakyrelu":		
			model.add(LeakyReLU(alpha=0.1))
		elif self.activation == "relu":
			model.add(Activation('relu'))
		### Batch-Normalization
		if self.batch_norm:
			model.add(BatchNormalization())
		### Max-Pooling
		model.add(MaxPooling2D((2, 2), padding='same'))

		### Convolution
		model.add(Conv2D(self.n_filters_layer5, self.filter_shape, activation='linear', kernel_initializer=initializer, padding='same'))
		### Activation
		if self.activation == "leakyrelu":		
			model.add(LeakyReLU(alpha=0.1))
		elif self.activation == "relu":
			model.add(Activation('relu'))
		### Batch-Normalization
		if self.batch_norm:
			model.add(BatchNormalization())
		### Max-Pooling
		model.add(MaxPooling2D((2, 2), padding='same'))

		### Dense Layer 1
		model.add(Flatten())
		model.add(Dense(self.denselayer_size, activation='linear', kernel_initializer=initializer))
		model.add(LeakyReLU(alpha=0.1))
		model.add(Dropout(self.dropout))

		### Output Layer
		model.add(Dense(self.n_classes, activation='softmax'))

		### Optimizer
		if self.optimizer == "Adam":
			opt = Adam(lr=self.l_rate)
		elif self.optimizer == "SGD":
			opt = SGD(lr=self.l_rate)

		### Compile the model
		model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=[categorical_accuracy])

		### Print the model summary
		model.summary()

		return model


	##################### Train Model #####################
	def TrainModel(self, X_train, y_train, X_val, y_val):

		########## If running wandb sweeps ##########
		if WANDB:
			wandb.run.name = "Scratch" + "_epoch_" + str(self.epochs) + "_bs_" + str(self.batch_size) + "_dls_" + str(self.denselayer_size) + "_lr_" + str(self.l_rate) + "_opt_" + self.optimizer + "_do_" + str(self.dropout) + "_act_" + self.activation + "_nfilt_" + str(self.n_filters) + "_fmul_" + str(self.filter_multiplier) + "_bn_" + str(self.batch_norm) + "_loss_" + self.loss

			history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, y_val), verbose=1, callbacks=[wandb.keras.WandbCallback(data_type="image", labels=class_names, save_model=False), EarlyStopping(monitor='val_categorical_accuracy', patience=3, verbose=1)])


		######### Normal run ########
		else:
			history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, y_val), verbose=1, callbacks=[EarlyStopping(monitor='val_categorical_accuracy', patience=3, verbose=1)])

			if os.path.exists('model.h5'):
				os.remove('model.h5')
			
			self.model.save('model.h5')
	
		return history


	##################### Test Model #####################
	def TestModel(self, X_test, y_test):
		
		test_eval = self.model.evaluate(X_test, y_test, verbose=0)
		print('Test loss:', test_eval[0])
		print('Test accuracy:', test_eval[1])
