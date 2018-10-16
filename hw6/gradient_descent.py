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
from numpy import exp
import pandas as pd
import matplotlib.pyplot as plt
from am207_utils import range_inc


# *************************************************************************************************
# 2.1. Construct an appropriate visualization of the loss function for the given data. 
# Use that visualization to verify that for  λ1=2.05384, λ2=0, the loss function L is minimized. 
# Your visualization should make note of this optima.

# Load the data
df = pd.read_csv('HW6_data.csv')
# Extract x1 and x2 
x1 = df['X_1'].values
x2 = df['X_2'].values
X_data = np.vstack([x1, x2]).T


def loss_func_abc(xs: np.ndarray, lambdas: np.ndarray, A: float, B: float, C: float) -> float:
    """
    General form of the loss function given in this problem.
    xs:       array of shape (n, 2)
    lambdas:  array of shape (2, 1)
    """
    # Unpack xs into 2 arrays of shape (n, 1)
    x1 = xs[:, 0]
    x2 = xs[:, 1]
    # Unpack lambdas into 2 scalars
    lambda1: float
    lambda2: float
    lambda1, lambda2 = lambdas
    # Term 1
    t1: float = A * (lambda2**2) * x2
    # Term 2
    t2: float = B * (lambda1**2) * x1

    # Difference of square x2^2 - x1^2
    ds: float = x2**2 - x1**2
    # Sum of squares lambda_1^2 + lambda_2^2
    ss: float = lambda1**2 + lambda2**2
    # Term 3
    t3: float = C * lambda1 * x1 * exp(ds * ss)
    # The function is the sum of these terms
    return np.sum(t1 + t2 + t3)

# The three numerical constants in the loss function
A: float = 0.000045
B: float = -0.000098
C: float = 0.003926


def loss_func(xs: np.ndarray, lambdas: np.ndarray) -> float:
    """
    Loss function given in this problem.  Binds parameters A, B, C to their given values.
    xs:       array of shape (n, 2)
    lambdas:  array of shape (2, 1)
    """
    # Reference to A, B, C and return loss function
    global A, B, C
    return loss_func_abc(xs, lambdas, A, B, C)


def L(lambda1: float, lambda2: float) -> float:
    """
    Loss function in terms of the parameters only; binds A, B, C and xs
    lambdas:  array of shape (2, 1)
    """
    # Reference to X_data
    global X_data
    # Combind lambdas into an array
    lambdas: np.ndarray = np.array([lambda1, lambda2])
    # Reference data and return loss
    return loss_func(X_data, lambdas)


# *************************************************************************************************
# 2.1 Visualize the loss function

# Minimum is given
lambda1_min = 2.05384
lambda2_min = 0.0
# lambdas_min = np.array([lambda1_min, lambda2_min])
loss_min = L(lambda1_min, lambda2_min)

# Grid of lambda1 and lambda2
grid_size: int = 200
lambda1_samp = np.linspace(-10, 10, grid_size)
lambda2_samp = np.linspace(-10, 10, grid_size)
lambda1_grid, lambda2_grid = np.meshgrid(lambda1_samp, lambda2_samp)
# Grid of the total loss function
if 'loss_grid' not in globals():
    loss_grid = np.zeros((grid_size, grid_size))
    for i, lambda1 in enumerate(lambda1_samp):
        for j, lambda2 in enumerate(lambda2_samp):
            loss_grid[i,j] = L(lambda1, lambda2)

# Plot the loss function
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Loss Function on Entire Data Set')
ax.set_xlabel(r'$\lambda_1$')
ax.set_ylabel(r'$\lambda_2$')
cs = ax.contour(lambda1_grid, lambda2_grid, loss_grid, linewidths=8)
ax.plot(lambda1_min, lambda2_min, label='Min', marker='o', markersize=12, linewidth=0, color='r')
fig.colorbar(cs, ax=ax)
ax.legend()
ax.grid()
plt.show()


# *************************************************************************************************
# 2.2. Choose an appropriate learning rate from [10, 1, 0.1, 0.01, 0.001, 0.0001] and 
# use that learning rate to implement gradient descent. Use your implementation to minimize L for the given data. 
# Your implementation should be stored in a function named gradient_descent.  
# gradient_descent should take the following parameters (n represents the number of data points):

def gradient_descent(lambda_init, X_data, step_size, max_iterations, precision, loss):
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
    # Initialize lambdas to lambda_init
    lambdas = lambda_init
    # Initialize history to have size max_iterations
    history = np.zeros(max_iterations+1)
    # Set shift size h for numerical derivatives of lambda1 and lambda2; use sqrt(machine_epsilon)
    h: float = 2**-26
    two_h: float = 2*h
    # Vectorized shifts to lambdas
    h_lam1 = np.array([h, 0])
    h_lam2 = np.array([0, h])
    # Initialize loss_prev to the loss on the initial parameter values
    loss_prev: float = loss(X_data, lambdas)
    history[0] = loss_prev
    # Perform up to max_iterations steps of gradient descent
    for i in range_inc(max_iterations):
        # Compute partial of loss w.r.t lambda1 and lambda2
        dL_dlam1 = (loss_func(X_data, lambdas + h_lam1) - loss_func(X_data, lambdas - h_lam1)) / two_h
        dL_dlam2 = (loss_func(X_data, lambdas + h_lam2) - loss_func(X_data, lambdas - h_lam2)) / two_h
        # Vector gradient dL_dlam
        grad = np.array([dL_dlam1, dL_dlam2])
        # Subtract a multiple of the gradient from lambdas
        lambdas = lambdas - step_size * grad
        # Compute the current loss
        loss_curr = loss(X_data, lambdas)
        # Save the current loss in the history
        history[i] = loss_curr
        # Compute the change in the loss function 
        loss_change = loss_prev - loss_curr
        # Update loss_prev
        loss_prev: float = loss_curr
        # Was the improvement below the precision? Then we can terminate
        if loss_change < precision:
            break
    # Prune history to the number of steps taken
    history = history[0:i+1]
    # Create the answer dictionary and return it
    gd = {'lambdas': lambdas, 
          'history' : history}
    return gd

# Run the gradient descent starting at [0, 0] with a learning rate of 0.01
lambda_init = np.array([0.0, 0.0])
step_size = 0.01
max_iterations=10000
precision=1e-12
gd = gradient_descent(lambda_init, X_data, step_size, max_iterations, precision, loss_func)
# Unpack answer
lambdas_gd = gd['lambdas']
history_gd = gd['history']
lambda1_gd, lambda2_gd = lambdas_gd
num_iters_gd = len(history_gd)-1
print(f'Completed gradient descent to precision {precision:0.2e} in {num_iters_gd} steps.')
print(f'lambda1 = {lambda1_gd:0.6f}, lambda2 = {lambda2_gd:0.6f}')


# *************************************************************************************************
# 2.3 For your implementation in 2.2, create a plot of loss vs iteration. 
# Does your descent algorithm comverge to the right values of  λ ? At what point does your implementation converge?

plot_n_gd = np.arange(num_iters_gd+1)
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Loss Function During Gradient Descent')
ax.set_xlabel('Iteration Number')
ax.set_ylabel('Loss Function')
ax.plot(plot_n_gd[0:100], history_gd[0:100], linewidth=4)
ax.grid()
plt.show()


# *************************************************************************************************
# 2.4. Choose an appropriate learning rate from [10, 1, 0.1, 0.01, 0.001, 0.0001] 
# and use that learning rate to implement stochastic gradient descent. Use your implementation to minimize 
# L for the given data. Your implementation should a stored in a function named stochastic_gradient_descent. 
# stochastic_gradient_descent should take the following parameters (n represents the number of data points):

def stochastic_gradient_descent(lambda_init, X_data, step_size, max_iterations, precision, loss):
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

    # Initialize lambdas to lambda_init
    lambdas = lambda_init
    # Initialize history to have size max_iterations
    history = np.zeros(max_iterations+1)
    # Set shift size h for numerical derivatives of lambda1 and lambda2; use sqrt(machine_epsilon)
    h: float = 2**-26
    two_h: float = 2*h
    # Vectorized shifts to lambdas
    h_lam1 = np.array([h, 0])
    h_lam2 = np.array([0, h])
    # Initialize loss_prev to the loss on the initial parameter values
    loss_prev: float = loss(X_data, lambdas)
    history[0] = loss_prev
    # Shape of data
    m, n = X_data.shape
    # Copy X so we don't re-order it when shuffling
    X = X_data.copy()
    # Perform up to max_iterations steps of gradient descent
    for epoch in range_inc(max_iterations):
        # Shuffle the data randomly for this epoch
        np.random.shuffle(X)
        # Iterate over individual points in X
        for i in range(m):
            # Extract the ith row
            row = X[i, :].reshape(-1,2)
            # Compute partial of loss w.r.t lambda1 and lambda2
            dL_dlam1 = (loss_func(row, lambdas + h_lam1) - loss_func(row, lambdas - h_lam1)) / two_h
            dL_dlam2 = (loss_func(row, lambdas + h_lam2) - loss_func(row, lambdas - h_lam2)) / two_h
            # Vector gradient dL_dlam
            grad = np.array([dL_dlam1, dL_dlam2])
            # Subtract a multiple of the gradient from lambdas
            lambdas = lambdas - step_size * grad
        # Compute the current loss
        loss_curr = loss(X_data, lambdas)
        # Save the current loss in the history
        history[epoch] = loss_curr
        # Compute the change in the loss function 
        loss_change = loss_prev - loss_curr
        # Update loss_prev
        loss_prev: float = loss_curr
        # Was the improvement below the precision? Then we can terminate
        if loss_change < precision:
            break
    # Prune history to the number of steps taken
    history = history[0:epoch+1]
    # Create the answer dictionary and return it
    sgd = {'lambdas': lambdas, 
          'history' : history}
    return sgd

# Run the gradient descent starting at [0, 0] with a learning rate of 0.01
lambda_init = np.array([0.0, 0.0])
step_size = 0.01
max_iterations=300
precision=1e-6
if 'sgd' not in globals():
    sgd = stochastic_gradient_descent(lambda_init, X_data, step_size, max_iterations, precision, loss_func)
# Unpack answer
lambdas_sgd = sgd['lambdas']
history_sgd = sgd['history']
lambda1_sgd, lambda2_sgd = lambdas_sgd
num_iters_sgd = len(history_sgd)-1
print(f'Completed stochastic gradient descent to precision {precision:0.2e} in {num_iters_sgd} steps.')
print(f'lambda1 = {lambda1:0.6f}, lambda2 = {lambda2:0.6f}')


# *************************************************************************************************
# 2.5 For your implementation in 2.4, create a plot of loss vs iteration. 
# Does your descent algorithm comverge to the right values of λ ? At what point does your implementation converge?
    
plot_n_sgd = np.arange(num_iters_sgd+1)
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Loss Function During Stochastic Gradient Descent')
ax.set_xlabel('Iteration Number')
ax.set_ylabel('Loss Function')
ax.plot(plot_n_sgd[0:100], history_sgd[0:100], linewidth=4)
ax.grid()
plt.show()


# *************************************************************************************************
# 2.6 Compare the average time it takes to update the parameter estimation in each iteration of the two 
# implementations. Which method is faster? Briefly explain why this result should be expected.

# See notebook

# *************************************************************************************************
# 2.7 Compare the number of iterations it takes for each algorithm to obtain an estimate accurate to 1e-3. 
# You may wish to set a cap for maximum number of iterations. 
# Which method converges to the optimal point in fewer iterations? Briefly explain why this result should be expected.


# *************************************************************************************************
# 2.8 Compare the performance of stochastic gradient descent on our loss function and dataset for the 
# following learning rates: [10, 1, 0.1, 0.01, 0.001, 0.0001]. 
# Based on your observations, briefly describe the effect of the choice of learning rate 
# on the performance of the algorithm.


# *************************************************************************************************
# 2.9 Using your implementation of gradient descent and stochastic gradient descent, document the behavior of your two
# algorithms for the following starting points, and for a number of stepsizes of your choice:
# (λ1,λ2) =(−2.47865,0)
# (λ1,λ2) =(−3,0)
# (λ1,λ2) =(−5,0)
# (λ1,λ2) =(−10,0)
# Construct a mathematical analysis of the loss function L to explain results of your descent 
# algorithms at different starting points.
