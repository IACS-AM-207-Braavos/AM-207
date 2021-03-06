"""
Harvard IACS AM 207
Homework 7
Problem 2

Michael S. Emanuel
Mon Oct 22 21:07:57 2018

Michael S. Emanuel
Wed Oct 24 00:15:40 2018
"""

import numpy as np
import matplotlib.pyplot as plt
# Import the classes from mnist that do all the actual work :)
from mnist import MNIST_Classifier, TwoLayerNetwork
from am207_utils import plot_style
from am207_utils import load_vartbl, save_vartbl
import time
from typing import Dict

# Set PLot Style
plot_style()


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
# Load persisted table of variables
fname: str = 'mnist_q2.pickle'
vartbl: Dict = load_vartbl(fname)


# *************************************************************************************************
# 2.1. Using a similar architecture as in Question 1 and the same training, validation and test sets, 
# build a PyTorch model for the multilayer perceptron. Use the tanh function as the non-linear activation function.

# Set key model parameters
num_hidden = 50
learning_rate = 0.1
weight_decay = 0.01
batch_size = 256
validation_size = 10000
epochs=20

# Handling - do we refit models in memory?
refit = False

# Train the logistic regression (i.e. softmax) model with the designated inputs
# Only run this block if the classifier not already in memory or the refit flag is set
try:    
    mnc = vartbl['mnc']
except:
    # Create a new instance of a logistic regression model
    model = TwoLayerNetwork(num_hidden)
    # Instantiate the MNIST Classifier- mnc for typability
    mnc = MNIST_Classifier(model=model,
                           learning_rate=learning_rate, 
                           weight_decay=weight_decay, 
                           batch_size=batch_size, 
                           validation_size=validation_size, 
                           epochs=epochs)

# Train the model
if not mnc.is_fit() or refit:
    # Fit the model
    mnc.fit(viz_val_loss=True)
    # Save the fitted model
    vartbl['mnc'] = mnc
    save_vartbl(vartbl, fname)
    

# *************************************************************************************************
# 2.2. The initialization of the weights matrix for the hidden layer must assure that the units (neurons) of the 
# perceptron operate in a regime where information gets propagated. For the  tanhtanh  function, you may find it 
# advisable to initialize with the interval  [a,a]. This is known as Xavier Initialization. 
# Use Xavier Initialization to initialize your MLP. 
# Feel free to use PyTorch's in-built Xavier Initialization methods.

# Please see __init__ method for class TwoLayerNetwork 

# *************************************************************************************************
# 2.3. Using  λ=0.01 to compare with Question 1, experiment with the learning rate (try 0.1 and 0.01 for example), 
# batch size (use 64, 128 and 256) and the number of units in your hidden layer (use between 25 and 200 units). 
# For what combination of these parameters do you obtain the highest validation accuracy? 
# You may want to start with 20 epochs for running time and experiment a bit to make sure that your 
# models reach convergence.


# Load the latest version of mncs if it's available
try:
    mncs = vartbl['mncs']
except:
    # Initialize an empty dictionary for mncs
    mncs = dict()

def test_parameters(mncs, learning_rates, batch_sizes, nums_hidden):
    """Iterate over all combinations of learning rate, batch size, and number of hidden units."""
    # Iteration counter
    i: int = 0
    # Compute total number of iterations
    iMax = len(learning_rates) * len(batch_sizes) * len(nums_hidden)
    skips = 0
    # Start the timer
    t0 = time.time()
    # Iterate over learning rates
    for learning_rate in learning_rates:
        # Iterate over batch sizes
        for batch_size in batch_sizes:
            # Iterate over number of hidden units
            for num_hidden in nums_hidden:
                # key for saving these parameter settings in the dictionary of models
                key = (learning_rate, batch_size, num_hidden)
                # Is this parameter setting already present in mncs? Then skip it.
                if key in mncs:
                    skips += 1
                    print(f'Already fit learning_rate={learning_rate}, batch_size={batch_size}, num_hidden={num_hidden}.')
                    continue
                # Create a two layer model with the given number of hidden units
                model = TwoLayerNetwork(num_hidden)
                # Create an MNIST classifier with the given learning rate and batch size
                mnc_curr = MNIST_Classifier(model=model,
                                       learning_rate=learning_rate, 
                                       weight_decay=weight_decay, 
                                       batch_size=batch_size, 
                                       validation_size=validation_size, 
                                       epochs=epochs)
                # Train this model; turn off visualization of loss by epoch until the end of training
                print(f'Training two layer network with learning_rate {learning_rate}, '
                      f'batch_size = {batch_size}, num_hidden={num_hidden}.')
                mnc_curr.fit(viz_val_loss=False)
                # Save this model to the dictionary mncs
                mncs[key] = mnc_curr
                # Save this to the vartbl
                vartbl['mncs'] = mncs
                save_vartbl(vartbl, fname)
                # Status update
                i += 1
                t1 = time.time()
                elapsed = (t1 - t0)
                projected = (iMax - skips - i) / i * elapsed
                print(f'Elapsed time {int(elapsed)}, projected remaining {int(projected)} (seconds).')

# *************************************************************************************************
# 2.3
# Range of parameter settings to try
learning_rates = [0.1, 0.01]
batch_sizes = [64, 128, 256]
nums_hidden = [25, 50, 100, 200]

# Test these parameters
test_parameters(mncs, learning_rates, batch_sizes, nums_hidden)

# Try all of these parameters, because this is what they told us to do...
# WARNING - this takes a long time to run, even if you have a GPU :(
# Runtime on Microsoft Surface Pro 4 laptop with Intel i7-6650 @ 2208 MHz, 2 cores
# was about 90 minutes  

# Find the combination with the best validation accuracy
def best_val(mnc) -> float:
    """Return the best validation score of a model"""
    val_scores = mnc.get_params('validation_losses')
    best_epoch = np.argmax(val_scores)
    best_score = val_scores[best_epoch]
    return best_score

# Dictionary with best score achieved by each model
best_scores = {key : best_val(mnc) for key, mnc in mncs.items()}
# Find the best key and the corresponding best mnc
best_key = max(best_scores, key=lambda k : best_scores[k])
best_mnc = mncs[best_key]
best_score = best_val(best_mnc)
best_lr, best_batch_sz, best_hidden = best_key
# Report results
print(f'The model with the best validation score had:')
print(f'Learning rate = {best_lr}')
print(f'Batch Size = {best_batch_sz}')
print(f'Hidden Units = {best_hidden}')
print(f'Validation Score = {best_score}')


# *************************************************************************************************
# 2.4. For your best combination plot the cross-entropy loss on the training set as a function of iteration.
losses = best_mnc.get_params("training_losses")
fig, axes = plt.subplots(nrows=1, ncols=epochs, figsize=(20,5), sharex=True, sharey=True)
for i in range(epochs):
    axes[i].plot(range(len(losses[i])), losses[i])
    axes[i].set_title(f'{i+1}')
    axes[i].set_xticks([])
    if i % 2 == 1:
        axes[i].axvspan(-10, 950, facecolor='gray', alpha=0.2)
plt.subplots_adjust(wspace=0)
plt.show()


# *************************************************************************************************
# 2.5. For your best combination use classification accuracy to evaluate how well your model is performing 
# on the validation set at the end of each epoch. Plot this validation accuracy as the model trains.

# The model training routine has a flag indicating whether or not to display the validation score
# live in training.  This was turned off when training all 24 models to avoid cluttering the screen
# with 24 models * 20 epochs = 480 intermediate charts
# Here is a single chart built after the fact showing the score vs. the epoch
best_mnc.viz_validation_loss(epochs-1)

# *************************************************************************************************
# 2.6. Select what you consider the best set of parameters and predict the labels of the test set. 
# Compare your predictions with the given labels. 
# What classification accuracy do you obtain on the training and test sets?

# The model above scored the best on the validation set and it has reasonable settings.
# Let's use that one and see how it does on the test set.

# Predict the labels on the training set
pred_train = np.array(best_mnc.predict('Train'))
labels_train = best_mnc.get_params('prediction_dataset_labels')
# Accuracy score on the test set
accuracy_train = best_mnc.score('Train')

# Predict the labels on the test set
pred_test = np.array(best_mnc.predict('Test'))
labels_test = best_mnc.get_params('prediction_dataset_labels')
# Accuracy score on the test set
accuracy_test = best_mnc.score('Test')

# Report the results
print(f'Accuracy on training and test data:')
print(f'Train: {accuracy_train*100:0.2f}%')
print(f'Test:  {accuracy_test*100:0.2f}%')


# *************************************************************************************************
# 2.7. How does your test accuracy compare to that of the logistic regression classifier in Question 1? 
# Compare best parameters for both models.

# See notebook

# *************************************************************************************************
# 2.8. What classes are most likely to be misclassified? Plot some misclassified training and test set images.

# Report errors on the test set
is_error = (pred_test != labels_test)
error_rate = np.zeros(10)
for d in range(0, 10):
    mask = (labels_test == d)
    count_d = np.sum(mask)
    error_d = np.sum(is_error[mask])
    error_rate[d] = error_d / count_d
# Get the three biggest error rates
# https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
top_errors = np.argpartition(error_rate, -3)[-3:][::-1]
print(f'Top three misclassified digits')
for d in top_errors:
    print(f'{d} : {error_rate[d]*100:0.2f}%')

# Plot misclassified training images
print(f'Misclassified Training Images:')
best_mnc.predict('Train')
best_mnc.viz_misclassified_images()

# Plot misclassified training images
print(f'Misclassified Test Images:')
best_mnc.predict('Test')
best_mnc.viz_misclassified_images()
