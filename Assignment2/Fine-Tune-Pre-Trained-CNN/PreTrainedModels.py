import numpy as np
import keras
import cv2
import os
import glob
from tqdm import tqdm
import pdb

from tensorflow.keras.applications import VGG16, VGG19, DenseNet201, MobileNet, ResNet50, InceptionResNetV2, InceptionV3, Xception
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy, binary_accuracy, binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping

from numpy.random import seed
from tensorflow.compat.v1 import set_random_seed

seed(1)
set_random_seed(42)


# To run wandb sweeps/run, change it to 1
WANDB = 0

if WANDB:
	import wandb
	from wandb.keras import WandbCallback
	wandb.init(config={"batch_size": 64, "l_rate": 0.001, "optimizer": "Adam", "epochs": 10, "denselayer_size": 128, "dropout": 0.5}, project="Deep-Learning-CNN")
	myconfig = wandb.config

class_names = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia']


WIDTH, HEIGHT, CHANNELS = 224, 224, 3

class CNN_Model():
	# Initialize the Hyperparameters
	def __init__(self, n_classes, n_filters, filter_size, filter_multiplier, var_n_filters, l_rate, epochs, optimizer, activation, loss, batch_size, initializer, data_augmentation, denselayer_size, batch_norm, train_model, model_version, dropout):

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
		self.model_version = model_version
		self.dropout = dropout

		self.model = self.InitializeModel()

	############ Initialize Model ############
	def InitializeModel(self):

		############ Popular CNN Architectures ############
		if self.model_version == 'VGG16':
			baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNELS)))
		elif self.model_version == 'VGG19':
			baseModel = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNELS)))
		elif self.model_version == 'DenseNet201':
			baseModel = DenseNet201(weights="imagenet", include_top=False, input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNELS)))
		elif self.model_version == 'MobileNet':
			baseModel = MobileNet(weights="imagenet", include_top=False, input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNELS)))
		elif self.model_version == 'ResNet50':
			baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNELS)))
		elif self.model_version == 'InceptionV3':
			baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNELS)))
		elif self.model_version == 'InceptionResNetV2':
			baseModel = InceptionResNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNELS)))
		elif self.model_version == 'Xception':
			baseModel = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNELS)))

		########### Adding classifier layer ###########
		headModel = baseModel.output
		headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
		headModel = Flatten(name="flatten")(headModel)
		headModel = Dense(self.denselayer_size, activation="relu")(headModel)
		headModel = Dropout(self.dropout)(headModel)
		headModel = Dense(self.n_classes, activation="softmax")(headModel)

		########### Freezing the layers in baseModel ###########
		for layer in baseModel.layers:
			layer.trainable = False

		########### Initializing the model ###########
		model = Model(inputs=baseModel.input, outputs=headModel)

		########### Display model summary ###########
		model.summary()

		########### Optimizer ###########
		if self.optimizer == "Adam":
			#opt = Adam(lr=self.l_rate, decay=self.l_rate/self.epochs)
			opt = Adam(lr=self.l_rate)
		elif self.optimizer == "SGD":
			opt = SGD(lr=self.l_rate)

		########### Compile the model ###########
		model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=[categorical_accuracy])

		return model


	##################### Train Model #####################
	def TrainModel(self, X_train, y_train, X_val, y_val):

		########## To run wandb ##########
		if WANDB:
			wandb.run.name = "Model_" + self.model_version + "_ep_" + str(self.epochs) + "_bs_" + str(self.batch_size) + "_dls_" + str(self.denselayer_size) + "_lr_" + str(self.l_rate) + "_opt_" + self.optimizer + "_do_" + str(self.dropout) + "_act_" + self.activation
	
			history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, y_val), verbose=1, callbacks=[wandb.keras.WandbCallback(data_type="image", labels=class_names, save_model=False), EarlyStopping(monitor='val_categorical_accuracy', patience=3, verbose=1)])

		########## Normal run without wandb ##########
		else:
			history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, y_val), verbose=1, callbacks=[EarlyStopping(monitor='val_categorical_accuracy', patience=3, verbose=1)])

			if os.path.exists('model.h5'):
				os.remove('model.h5')
			
			self.model.save('model.h5')

		return history

	##################### Test Model #####################
	def TestModel(self, X_test, y_test):
		
		test_eval = self.model.evaluate(X_test, y_test, verbose=0)
		return test_eval
