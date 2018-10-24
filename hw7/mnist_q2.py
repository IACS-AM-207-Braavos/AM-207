"""
Harvard IACS AM 207
Homework 7
Problem 2

Michael S. Emanuel
Mon Oct 22 21:07:57 2018

Michael S. Emanuel
Wed Oct 24 00:15:40 2018
"""

# *************************************************************************************************
# Question 2: MNIST MLP! Find out what that means to me. MNIST MLP! Take care, TCB!
# *************************************************************************************************

# The multilayer perceptron can be understood as a logistic regression classifier in which the input is first 
# transformed using a learnt non-linear transformation. The non-linear transformation is often chosen to be either 
# the logistic function or the tanh function or the RELU function, and its purpose is to project the data into a 
# space where it becomes linearly separable. 
# The output of this so-called hidden layer is then passed to the logistic regression graph 
# that we have constructed in the first problem.

# We'll construct a model with 1 hidden layer. That is, you will have an input layer, then a hidden layer with the 
# nonlinearity, and finally an output layer with cross-entropy loss 
# (or equivalently log-softmax activation with a negative log likelihood loss).


# *************************************************************************************************
# 2.1. Using a similar architecture as in Question 1 and the same training, validation and test sets, 
# build a PyTorch model for the multilayer perceptron. Use the  tanhtanh  function as the non-linear activation function.


# *************************************************************************************************
# 2.2. The initialization of the weights matrix for the hidden layer must assure that the units (neurons) of the 
# perceptron operate in a regime where information gets propagated. For the  tanhtanh  function, you may find it 
# advisable to initialize with the interval  [a,a]. This is known as Xavier Initialization. 
# Use Xavier Initialization to initialize your MLP. 
# Feel free to use PyTorch's in-built Xavier Initialization methods.


# *************************************************************************************************
# 2.3. Using  λ=0.01 to compare with Question 1, experiment with the learning rate (try 0.1 and 0.01 for example), 
# batch size (use 64, 128 and 256) and the number of units in your hidden layer (use between 25 and 200 units). 
# For what combination of these parameters do you obtain the highest validation accuracy? 
# You may want to start with 20 epochs for running time and experiment a bit to make sure that your 
# models reach convergence.


# *************************************************************************************************
# 2.4. For your best combination plot the cross-entropy loss on the training set as a function of iteration.


# *************************************************************************************************
# 2.5. For your best combination use classification accuracy to evaluate how well your model is performing 
# on the validation set at the end of each epoch. Plot this validation accuracy as the model trains.


# *************************************************************************************************
# 2.6. Select what you consider the best set of parameters and predict the labels of the test set. 
# Compare your predictions with the given labels. 
# What classification accuracy do you obtain on the training and test sets?


# *************************************************************************************************
# 2.7. How does your test accuracy compare to that of the logistic regression classifier in Question 1? 
# Compare best parameters for both models.


# *************************************************************************************************
# 2.8. What classes are most likely to be misclassified? Plot some misclassified training and test set images.