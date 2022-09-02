import numpy as np
import cv2
import glob
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

from numpy.random import seed
seed(1)


def ReadData(WIDTH, HEIGHT, CHANNELS, train_data_dir, test_data_dir, read_data=True):
	################# Reading Data #################

	if read_data:

		inaturalist_labels_dict = {
		    'Amphibia': 0,
		    'Animalia': 1,
		    'Arachnida': 2,
		    'Aves': 3,
		    'Fungi': 4,
		    'Insecta': 5,
		    'Mammalia': 6,
		    'Mollusca': 7,
		    'Plantae': 8,
		    'Reptilia': 9,
		}
	
		######### Read Train Data #########
		inaturalist_train_dict = {
		    'Amphibia': list(glob.glob(train_data_dir + 'Amphibia/*')),
		    'Animalia': list(glob.glob(train_data_dir + 'Animalia/*')),
		    'Arachnida': list(glob.glob(train_data_dir + 'Arachnida/*')),
		    'Aves': list(glob.glob(train_data_dir + 'Aves/*')),
		    'Fungi': list(glob.glob(train_data_dir + 'Fungi/*')),
		    'Insecta': list(glob.glob(train_data_dir + 'Insecta/*')),
		    'Mammalia': list(glob.glob(train_data_dir + 'Mammalia/*')),
		    'Mollusca': list(glob.glob(train_data_dir + 'Mollusca/*')),
		    'Plantae': list(glob.glob(train_data_dir + 'Plantae/*')),
		    'Reptilia': list(glob.glob(train_data_dir + 'Reptilia/*')),
		}

		X, y = [], []

		for species_name, images in inaturalist_train_dict.items():
			print("##################### " + species_name + " #####################")
			for image in tqdm(images, total=len(images)):
				img = cv2.imread(str(image))
				resized_img = cv2.resize(img,(WIDTH, HEIGHT))
				X.append(resized_img)
				y.append(inaturalist_labels_dict[species_name])

		######### Convert to Numpy Array #########
		X = np.array(X)
		y = np.array(y)

		######### Train-Test Split #########
		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

		######### Reshape #########
		X_train = X_train.reshape(X_train.shape[0], WIDTH, HEIGHT, 3)
		X_val = X_val.reshape(X_val.shape[0], WIDTH, HEIGHT, 3)
		y_train = y_train.reshape(len(y_train), 1)
		y_val = y_val.reshape(len(y_val), 1)

		######### Convert to one-hot vector #########
		y_train = to_categorical(y_train)
		y_val = to_categorical(y_val)


		######### Read Test Data #########
		inaturalist_test_dict = {
		    'Amphibia': list(glob.glob(test_data_dir + 'Amphibia/*')),
		    'Animalia': list(glob.glob(test_data_dir + 'Animalia/*')),
		    'Arachnida': list(glob.glob(test_data_dir + 'Arachnida/*')),
		    'Aves': list(glob.glob(test_data_dir + 'Aves/*')),
		    'Fungi': list(glob.glob(test_data_dir + 'Fungi/*')),
		    'Insecta': list(glob.glob(test_data_dir + 'Insecta/*')),
		    'Mammalia': list(glob.glob(test_data_dir + 'Mammalia/*')),
		    'Mollusca': list(glob.glob(test_data_dir + 'Mollusca/*')),
		    'Plantae': list(glob.glob(test_data_dir + 'Plantae/*')),
		    'Reptilia': list(glob.glob(test_data_dir + 'Reptilia/*')),
		}

		X_test, y_test = [], []

		for species_name, images in inaturalist_test_dict.items():
			print("##################### " + species_name + " #####################")
			for image in tqdm(images, total=len(images)):
				img = cv2.imread(str(image))
				resized_img = cv2.resize(img,(WIDTH, HEIGHT))
				X_test.append(resized_img)
				y_test.append(inaturalist_labels_dict[species_name])

		######### Convert to Numpy Array #########
		X_test = np.array(X_test)
		y_test = np.array(y_test)

		######### Reshape #########
		X_test = X_test.reshape(X_test.shape[0], WIDTH, HEIGHT, 3)
		y_test = y_test.reshape(len(y_test), 1)

		######### Convert to one-hot vector #########
		y_test = to_categorical(y_test)
		
		np.savez('data.npz', X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val, X_test=X_test, y_test=y_test)

	###### If the data is already saved #####
	else:

		################# Reading Stored Data #################
		data = np.load('/cbr/saish/Datasets/data.npz')
		X_train, X_val, X_test, y_train, y_val, y_test = data['X_train'], data['X_val'], data['X_test'], data['y_train'], data['y_val'], data['y_test']

		X_train = X_train.astype('float32')
		X_val = X_val.astype('float32')
		X_test = X_test.astype('float32')
		X_train = X_train / 255.
		X_val = X_val / 255.
		X_test = X_test / 255.

	return X_train, X_val, X_test, y_train, y_val, y_test
