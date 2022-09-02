import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import pdb

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam
import keras

from tensorflow.keras.models import load_model


#Setting the image apth and the last conv layer for VGG19
IMAGE_PATH = '/cbr/saish/PhD/DL/Assignment_2/CNN-From-Scratch/aves_bf35.jpg'
LAYER_NAME='block5_conv4'

#Load the image
img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))

# Displaying the original image
plt.axis("off")
plt.imshow(img)
plt.savefig("op.png")

# Preprocess the image using vgg19 preprocess function
img =  tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(img, axis=0)

# Load Model
model = load_model('model.h5')

# Model Summary
model.summary()

# Layer name -- conv5 layer
LAYER_NAME = 'conv2d_4'

#create a model till  last convolutional layers to have the best compromise between high-level semantics and detailed spatial information
gb_model = tf.keras.models.Model(
    inputs = [model.inputs],    
    outputs = [model.get_layer(LAYER_NAME).output]
)
layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer,'activation')]

# Define custom gradient
@tf.custom_gradient
def guidedRelu(x):
  def grad(dy):
    return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
  return tf.nn.relu(x), grad

# Add guided relu activation to all the layers
for layer in layer_dict:
  if layer.activation == tf.keras.activations.relu:
    layer.activation = guidedRelu

# Get gradients
with tf.GradientTape() as tape:
 inputs = tf.cast(x, tf.float32)
 tape.watch(inputs)
 outputs = gb_model(inputs)[0]
grads = tape.gradient(outputs,inputs)[0]

pdb.set_trace()

#Visualizing the guided back prop
guided_back_prop =grads.numpy()
gb_viz = np.dstack((guided_back_prop[:, :, 0], guided_back_prop[:, :, 1], guided_back_prop[:, :, 2],))       
gb_viz -= np.min(gb_viz)
gb_viz /= gb_viz.max()
    
plt.imshow(gb_viz)
plt.axis("off")
plt.savefig("gbp.png")
