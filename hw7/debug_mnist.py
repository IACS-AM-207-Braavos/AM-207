"""
Michael S. Emanuel
Tue Oct 23 21:45:46 2018
"""

import numpy as np
# import torch.nn.functional as F
from mnist import MNIST_Classifier, LogisticRegression, SoftmaxRegression, TwoLayerNetwork


# Set key model parameters
learning_rate = 0.1
weight_decay = 0.001
batch_size = 256
validation_size = 10000
epochs=2

## Define our model 
# Create a new instance of a logistic regression model
# model = LogisticRegression() 
model = SoftmaxRegression()
# Instance of softmax r
# Instantiate the MNIST Classifier- mnc for typability
mnc = MNIST_Classifier(model=model,
                       learning_rate=learning_rate, 
                       weight_decay=weight_decay, 
                       batch_size=batch_size, 
                       validation_size=validation_size, 
                       epochs=epochs)

## We defined a number of variables in our constructor -- let's reclaim them here
optimizer=mnc.get_params("optimizer")
model=mnc.get_params("model")
epochs=mnc.get_params("epochs")
criterion=mnc.get_params("criterion")
train_loader=mnc.get_params("train_loader")

training_size = mnc.get_params('train_dataset').train_data.size(0)


iterations = int(np.ceil(training_size/mnc.get_params("batch_size")))

## We need something to keep track of our losses
losses = np.zeros((epochs, iterations)) 
# Also track the losses at the end of every epoch
losses_val = np.zeros(epochs)


train_iter = train_loader.__iter__()
batch_index = 0
inputs, labels = next(train_iter)
inputs = inputs.view(-1, 28*28)
optimizer.zero_grad()
# outputs = model(inputs)

smr = SoftmaxRegression()
x = inputs
z1 = smr.layer_1_linear(x)
y = smr.layer_2_softmax(z1)

tln = TwoLayerNetwork(50)
x = inputs
z1 = tln.layer_1_linear(x)
a1 = tln.layer_1_activation(z1) 
y = tln.layer_2_linear(a1)
