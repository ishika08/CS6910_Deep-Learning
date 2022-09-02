**Implementation of Feedforward Neural Network using NumPy library**

[Link to the wandb.ai report](https://wandb.ai/saish/Deep-Learning/reports/CS6910-Assignment-1--Vmlldzo1MzI2OTE)

**Usage**
```
usage: python3 main.py [-h] 
		[--n_classes N_CLASSES] 
		[--n_hlayers N_HLAYERS]
		[-l LAYER_SIZES [LAYER_SIZES ...]]
		[--l_rate L_RATE]
		[--epochs EPOCHS] --optimizer OPTIMIZER
		[--activation ACTIVATION]
		[--loss LOSS]               	
		[--output_activation OUTPUT_ACTIVATION]      		
		[--batch_size BATCH_SIZE] 
		[--initializer INITIALIZER]               	
		[--hlayer_size HLAYER_SIZE]
```

Hyperparameter | Values/Usage
-------------------- | --------------------
n_classes | 10
n_hlayers | 3, 4, 5
-l (variable length layers) | 784 128 64 32 10
epochs | 5, 10
activation | 'sigmoid', 'relu', 'tanh'
loss | 'cross_entropy', 'squared_error'
output_activation | 'softmax'
batch_size | 16, 32, 64
initializer | 'xavier', 'random'
hlayer_size | 32, 64, 128


**To run the code**

1. Set the hyperparameter configuration in run.sh file
2. Run `bash run.sh`

**Files**

* run.sh -- to run the code
* main.py -- main funtion, loads dataset
* FeedForwardNN.py -- contains class definition for a FFNN model
