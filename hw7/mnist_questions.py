"""
Harvard IACS AM 207
Homework 7
Problem 1

Michael S. Emanuel
Mon Oct 22 21:07:57 2018
"""

import numpy as np
# Import the classes from mnist that do all the actual work :)
from mnist import MNIST_Classifier, LogisticRegression
from am207_utils import plot_style
from am207_utils import load_vartbl, save_vartbl
from typing import Dict

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

# *************************************************************************************************
# Load persisted table of variables
fname: str = 'mnist.pickle'
vartbl: Dict = load_vartbl(fname)

# Train the logistic regression (i.e. softmax) model with the designated inputs
# Set key model parameters
learning_rate = 0.1
weight_decay = 0.01
batch_size = 256
validation_size = 10000
epochs=10
refit = False

## Only run this block if the classifier not already in memory or the refit flag is set
try:    
    mnc = vartbl['mnc']
except:
    # Create a new instance of a logistic regression model
    model = LogisticRegression()
    # Instantiate the MNIST Classifier- mnc for typability
    mnc = MNIST_Classifier(model=model,
                           learning_rate=learning_rate, 
                           weight_decay=weight_decay, 
                           batch_size=batch_size, 
                           validation_size=validation_size, 
                           epochs=epochs)

# *************************************************************************************************
# 1.1. Plot 10 sample images from the MNIST dataset (to develop intuition for the feature space).

## Plot sample images
mnc.viz_training_images()

# *************************************************************************************************
# 1.2. Currently the MNIST dataset in Torchvision allows a Train/Test split. 
# Use PyTorch dataloader functionality to create a Train/Validate/Test split of 50K/10K/10K samples.

# Hint: Lab described a way to do it keeping within the MNIST DataLoader workflow: 
# the key is to pass a SubsetRandomSampler to DataLoader

# See REVISED version of method load_data() in class MNIST_Classifier

# *************************************************************************************************
# 1.3. Construct a softmax formulation in PyTorch of multinomial logistic regression with Cross Entropy Loss.

# No changes required to code in the lab; created explicit SoftmaxRegression for fun
# See class LogisticRegression

# *************************************************************************************************
# 1.4. Train your model using SGD to minimize the cost function. 
# Use as many epochs as you need to achieve convergence.
if not mnc.is_fit() or refit:
    # Fit the model
    mnc.fit(viz_val_loss=True)
    # Save the fitted model
    vartbl['mnc'] = mnc
    save_vartbl(vartbl, fname)

# *************************************************************************************************
# 1.5. Plot the cross-entropy loss on the training set as a function of iteration.
mnc.viz_training_loss(epochs)

# *************************************************************************************************
# 1.6a. Using classification accuracy, evaluate how well your model is performing on the validation set 
# at the end of each epoch. Plot this validation accuracy as the model trains.

# Plot done interactively as model trains
# Here is one version of it
mnc.viz_validation_loss(epochs-1)

# *************************************************************************************************
# 1.6b. Duplicate this plot for some other values of the regularization parameter λ. 
# When should you stop the training for each of the different values of λ? 
# Give an approximate answer supported by using the plots.

try:
    mncs_wd = vartbl['mncs_wd']
    weight_decays = list(mncs_wd.keys())
    num_wd = len(weight_decays)
    # Iterate over the weight decays and print the validation loss
    for mnc_wd in mncs_wd.values():
        mnc_wd.viz_validation_loss(epochs-1)
except:
    # Test seven values of lambda: the original value of 0.01, plus three log-spaced numbers on either side
    weight_decays = [0.00031, 0.001, 0.0031, 0.01, 0.031, 0.1, 0.31]
    num_wd = len(weight_decays)
    mncs_wd = dict()
    # Original weight decay to avoid refitting the same model
    wd_orig = mnc.get_params('weight_decay')
    # Iterate over the weight decays
    for weight_decay in weight_decays:
        # Don't duplicate weight decay we already ran
        if weight_decay == wd_orig:
            mncs_wd[weight_decay] = mnc
            mnc.viz_validation_loss(epochs-1)
            continue
        # Instantiate a model with this weight decay
        model = LogisticRegression()
        mnc_wd =  MNIST_Classifier(model=model,
                               learning_rate=learning_rate, 
                               weight_decay=weight_decay, 
                               batch_size=batch_size, 
                               validation_size=validation_size, 
                               epochs=epochs)
        # Train this model; turn off visualization of loss by epoch until the end of training
        print(f'Training simple logistic regression model with weight decay = {weight_decay}.')
        mnc_wd.fit(viz_val_loss=False)
        # Save this model to the dictionary mncs_wd
        mncs_wd[weight_decay] = mnc_wd
        # Save this to the vartbl
        vartbl['mncs_wd'] = mncs_wd
        save_vartbl(vartbl, fname)


# *************************************************************************************************
# 1.7. Select what you consider the best regularization parameter and predict the labels of the test set. 
# Compare your predictions with the given labels. 
# What classification accuracy do you obtain on the training and test sets?

# Use weight_decay = 0.001 because it has the best scores (in general, not just at the peak or end)
wd_best = 0.001
mnc_best = mncs_wd[wd_best]


# Predict the labels on the training set
pred_train = np.array(mnc_best.predict('Train'))
labels_train = mnc_best.get_params('prediction_dataset_labels')
# Accuracy score on the test set
accuracy_train = mnc_best.score('Train')

# Predict the labels on the test set
pred_test = np.array(mnc_best.predict('Test'))
labels_test = mnc_best.get_params('prediction_dataset_labels')
# Accuracy score on the test set
accuracy_test = mnc_best.score('Test')

# Report the results
print(f'Accuracy on training and test data:')
print(f'Train: {accuracy_train*100:0.2f}%')
print(f'Test:  {accuracy_test*100:0.2f}%')


# *************************************************************************************************
# 1.8. What classes are most likely to be misclassified? 
# Plot some misclassified training and test set images.

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
mnc_best.predict('Train')
mnc_best.viz_misclassified_images()

# Plot misclassified training images
print(f'Misclassified Test Images:')
mnc_best.predict('Test')
mnc_best.viz_misclassified_images()


