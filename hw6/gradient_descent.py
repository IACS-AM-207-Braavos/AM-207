"""
Harvard IACS AM 207
Homework 6
Problem 2

Michael S. Emanuel
Mon Oct 15 22:30:43 2018
"""

# *************************************************************************************************
# Question 2: Mr. Poe Writes of Gradient Descent Into the Maelström
# *************************************************************************************************

# Suppose you are building a pricing model for laying down telecom cables over a geographical region. 
# You construct a pricing model that takes as input a pair of coordinates,  (x1,x2) and based upon two parameters  
# λ1, λ2 predicts the loss in revenue corresponding to laying the cables at the inputed location. 
# Your pricing model is described by the following equation:
# L(x1,x2 | λ1,λ2) = 0.000045 λ2^2 y −0.000098λ1^2 x1 + 0.003926 λ1 x1 exp{(x2^2 − x1^2)(λ1^2 + λ2^2)}

 
# We've provided you some data contained in the file HW6_data.csv. 
# This data represents a set of coordinates configured on the curve  x22−x21=−0.1x22−x12=−0.1 . 
# Your goal for this problem is to find the parameters λ1, λ2 that minimize the net loss over the entire dataset.

import numpy as np
import pandas as pd


# *************************************************************************************************
# 2.1. Construct an appropriate visualization of the loss function for the given data. 
# Use that visualization to verify that for  λ1=2.05384, λ2=0, the loss function L is minimized. 
# Your visualization should make note of this optima.

# Load the data
df = pd.read_csv('HW6_data.csv')



# *************************************************************************************************
# 2.2. Choose an appropriate learning rate from [10, 1, 0.1, 0.01, 0.001, 0.0001] and 
# use that learning rate to implement gradient descent. Use your implementation to minimize L for the given data. 
# Your implementation should be stored in a function named gradient_descent.  
# gradient_descent should take the following parameters (n represents the number of data points):

def gradient_descent():
    """
    Gradient Descent algorithm specialized for this problem.

    INPUTS:
    ======
    lambda_init    -- a numpy array with shape (2, 1) containing the initial value for λ1 and λ2 
    X_data         -- an numpy array with shape (n, 2) containing the data coordinates used in your loss function
    step_size      -- a float containing the step-size/learning rate used in your algorithm
    scale          -- a float containing the factor by which you'll scale your step_size 
                     (or alternatively your loss) in the algorithm
    max_iterations -- an integer containing a cap on the number of iterations for which you'll let your algorithm run
    precision      -- a float containing the difference in loss between consecutive iterations below which 
                      you'll stop the algorithm
    loss           -- a function (or lambda function) that takes in the following parameters and returns a float 
                      with the results of calculating the loss function for our data at λ1 and λ2 
            lambdas        -- a numpy array with shape (2, 1) containing  λ1 and λ2 
            X_data         -- the same as the parameter X_data for gradient_descent
    
    RETURNS:
    =======
    Dictionary with the following keys (n_iterations represents the total number of iterations):
    'lambdas' -- the associated value is a numpy array with shape (2,1) containing 
                 the optimal λ's found by the algorithm
    'history' -- the associated value is a numpy array with shape (n_iterations,) containing a 
                 history of the calculated value of the loss function at each iteration        
    """
    pass


# *************************************************************************************************
# 2.3 For your implementation in 2.2, create a plot of loss vs iteration. 
# Does your descent algorithm comverge to the right values of  λ ? At what point does your implementation converge?
    


# *************************************************************************************************
# 2.4. Choose an appropriate learning rate from [10, 1, 0.1, 0.01, 0.001, 0.0001] 
# and use that learning rate to implement stochastic gradient descent. Use your implementation to minimize 
# L for the given data. Your implementation should a stored in a function named stochastic_gradient_descent. 
# stochastic_gradient_descent should take the following parameters (n represents the number of data points):

def stochastic_gradient_descent():
    """
    Stochastic Gradient Descent algorithm specialized for this problem.

    INPUTS
    ======
    lambda_init    -- a numpy array with shape (2, 1) containing the initial value for λ1 and λ2 
    X_data         -- an numpy array with shape (n, 2) containing the data coordinates used in your loss function
    step_size      -- a float containing the step-size/learning rate used in your algorithm
    scale          -- a float containing the factor by which you'll scale your step_size 
                     (or alternatively your loss) in the algorithm
    max_iterations -- an integer containing a cap on the number of iterations for which you'll let your algorithm run
    precision      -- a float containing the difference in loss between consecutive iterations below which 
                      you'll stop the algorithm
    loss           -- a function (or lambda function) that takes in the following parameters and returns a float 
                      with the results of calculating the loss function for our data at λ1 and λ2 
            lambdas        -- a numpy array with shape (2, 1) containing  λ1 and λ2 
            X_data         -- the same as the parameter X_data for gradient_descent
    
    RETURNS:
    =======
    Dictionary with the following keys (n_iterations represents the total number of iterations):
    'lambdas' -- the associated value is a numpy array with shape (2,1) containing 
                 the optimal λ's found by the algorithm
    'history' -- the associated value is a numpy array with shape (n_iterations,) containing a 
                 history of the calculated value of the loss function at each iteration        
    """
    pass



# *************************************************************************************************
# 2.5 For your implementation in 2.4, create a plot of loss vs iteration. 
# Does your descent algorithm comverge to the right values of λ ? At what point does your implementation converge?
    

# *************************************************************************************************
# 2.6 Compare the average time it takes to update the parameter estimation in each iteration of the two implementations. Which method is faster? Briefly explain why this result should be expected.


# *************************************************************************************************
# 2.7 Compare the number of iterations it takes for each algorithm to obtain an estimate accurate to 1e-3. You may wish to set a cap for maximum number of iterations. Which method converges to the optimal point in fewer iterations? Briefly explain why this result should be expected.


# *************************************************************************************************
# 2.8 Compare the performance of stochastic gradient descent on our loss function and dataset for the following learning rates: [10, 1, 0.1, 0.01, 0.001, 0.0001]. Based on your observations, briefly describe the effect of the choice of learning rate on the performance of the algorithm.


# *************************************************************************************************
# 2.9 Using your implementation of gradient descent and stochastic gradient descent, document the behavior of your two
# algorithms for the following starting points, and for a number of stepsizes of your choice:
# (λ1,λ2) =(−2.47865,0)
# (λ1,λ2) =(−3,0)
# (λ1,λ2) =(−5,0)
# (λ1,λ2) =(−10,0)
# Construct a mathematical analysis of the loss function L to explain results of your descent 
# algorithms at different starting points.
