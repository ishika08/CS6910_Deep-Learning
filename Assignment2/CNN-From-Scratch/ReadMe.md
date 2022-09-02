<h2>Training a Convolutional Neural Network from Scratch</h2>

[Link to the wandb.ai report](https://wandb.ai/saish/Deep-Learning-CNN/reports/CS6910-Assignment-2---Vmlldzo2MDQ4NDA?accessToken=0d2a802xore8clx738gb2wuytbi54q9lyh6g4rlwxpt4bvs3d57qo3gc7uzisgzs)

**Usage**

```
usage: main.py [-h] [--n_classes N_CLASSES] [--n_filters N_FILTERS]
               [--filter_multiplier FILTER_MULTIPLIER]
               [--filter_size FILTER_SIZE]
               [-l VAR_N_FILTERS [VAR_N_FILTERS ...]] [--l_rate L_RATE]
               [--epochs EPOCHS] [--optimizer OPTIMIZER]
               [--activation ACTIVATION] [--loss LOSS]
               [--batch_size BATCH_SIZE] [--initializer INITIALIZER]
               [--data_augmentation DATA_AUGMENTATION]
               [--denselayer_size DENSELAYER_SIZE] [--batch_norm BATCH_NORM]
               [--train_model TRAIN_MODEL] [--dropout DROPOUT]

```

Hyperparameter | Values/Usage
-------------------- | --------------------
n_classes | 10
filter_multiplier | 1, 0.5, 2
-l | to pass variable number of filters for each layer
epochs | 5, 10
optimizer | "Adam", "SGD"
activation | "relu", "leakyrelu"
loss | "categorical_crossentropy"
batch_size | 64, 32
initializer | "orthogonal"
data_augmentation | True, False
denselayer_size | 64, 128
train_model | "Yes", "No"
dropout | 0.2, 0.4, 0.5


**To run the code**

1. Set the hyperparameter configuration in run.sh file
2. Run `bash run.sh`

**Files**

* run.sh -- to run the code
* main.py -- main funtion
* LoadData.py -- to load the data
* CNNModel.py -- to initialize and train the model
* GuidedBackprop.py -- for performing guided backpropagation
* config.py -- to set up configuration
* sweep.yaml -- for configuring hyperparameters to run sweeps
