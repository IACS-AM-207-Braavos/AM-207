"""
Harvard IACS AM 207
Homework 7
Problem 1

Michael S. Emanuel
Mon Oct 22 21:07:57 2018
"""

import numpy as np
import scipy.stats
import scipy.special

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import cm
import pandas as pd

## Standard boilerplate to import torch and torch related modules
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

from pytorch_lab import MNIST_Logistic_Regression
from am207_utils import plot_style

# Set PLot Style
plot_style()

# *************************************************************************************************
# Question 1: Mon pays c'est l'MNIST. Mon cœur est brise de Logistic Regression.
# *************************************************************************************************

# The MNIST dataset is one of the classic datasets in Machine Learning and is often one of the first datasets against 
# which new classification algorithms test themselves. It consists of 70,000 images of handwritten digits, 
# each of which is 28x28 pixels. You will be using PyTorch to build a handwritten digit classifier that you will 
# train, validate, and test with MNIST.

# Your classifier MUST implement a multinomial logistic regression model (using softmax). 
# It will take as input an array of pixel values in an image and output the images most likely digit label (i.e. 0-9). 
# You should think of the pixel values as features of the input vector.

# Using the softmax formulation, your PyTorch model should computes the cost function using an L2 regularization 
# approach (see optim.SGD in PyTorch or write your own cost function) and minimize the resulting cost function using 
# mini-batch stochastic gradient descent. We provided extensive template code in lab.

# Construct and train your classifier using a batch size of 256 examples, 
# a learning rate η=0.1, and a regularization factor λ=0.01.

# Set key model parameters
learning_rate = 0.1
weight_decay = 0.001
batch_size = 256
validation_size = 10000
epochs=10

## Define our model 
if 'MLR' not in globals():
    MLR = MNIST_Logistic_Regression(learning_rate=learning_rate,
                                    weight_decay=weight_decay,
                                    batch_size=batch_size,
                                    validation_size=validation_size,
                                    epochs=epochs)

# *************************************************************************************************
# 1.1. Plot 10 sample images from the MNIST dataset (to develop intuition for the feature space).

## Plot sample images
MLR.viz_training_images()

# *************************************************************************************************
# 1.2. Currently the MNIST dataset in Torchvision allows a Train/Test split. 
# Use PyTorch dataloader functionality to create a Train/Validate/Test split of 50K/10K/10K samples.

# Hint: Lab described a way to do it keeping within the MNIST DataLoader workflow: 
# the key is to pass a SubsetRandomSampler to DataLoader

# Done in revised version of load_data()

# *************************************************************************************************
# 1.3. Construct a softmax formulation in PyTorch of multinomial logistic regression with Cross Entropy Loss.

# No changes required to code in the lab (?)

# *************************************************************************************************
# 1.4. Train your model using SGD to minimize the cost function. 
# Use as many epochs as you need to achive convergence.
if not MLR.is_fit():
    MLR.fit()

# *************************************************************************************************
# 1.5. Plot the cross-entropy loss on the training set as a function of iteration.
MLR.viz_training_loss(epochs)

# *************************************************************************************************
# 1.6a. Using classification accuracy, evaluate how well your model is performing on the validation set 
# at the end of each epoch. Plot this validation accuracy as the model trains.

# Plot done as model trains

# *************************************************************************************************
# 1.6b. Duplicate this plot for some other values of the regularization parameter λ. 
# When should you stop the training for each of the different values of λ? 
# Give an approximate answer supported by using the plots.

lams = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

# Lambda = 0.000: stop at end of epoch 1; accuracy = 
# Lambda = 0.001: stop at end of epoch 6; accuracy = 91.2
# Lambda = 0.005: stop at end of epoch 3; accuracy = 91.2
# Lambda = 0.01: stop at end of epoch 5; accuracy = 
# Lambda = 0.05: stop at end of epoch 5; accuracy = 90.3
# Lambda = 0.10: 
# Lambda = 0.50: 
# Lambda = 1.00: 

# *************************************************************************************************
# 1.7. Select what you consider the best regularization parameter and predict the labels of the test set. 
# Compare your predictions with the given labels. 
# What classification accuracy do you obtain on the training and test sets?



# *************************************************************************************************
# 1.8. What classes are most likely to be misclassified? 
# Plot some misclassified training and test set images.


# *************************************************************************************************

