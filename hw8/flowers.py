"""
Harvard AM 207
Homework 8

Michael S. Emanuel
Thu Nov  1 16:08:08 2018
"""


from numpy import log
from typing import Tuple
# Type alias
data_type = Tuple[int, int, int, int]


def log_like(Y: data_type, theta: float) -> float:
    """The log likelihood function of this data given a parameter setting theta"""
    # Unpack Y
    (y1, y2, y3, y4) = Y
    # Return the observed data likelihood from question 1.1
    return y1 * log(2.0 + theta) + (y2 + y3) * log(1.0 - theta) + y4 * log(theta)


def z_bar(theta: float, Y: data_type) -> float:
    """The E step of the EM algorithm.  Compute expected value of z from theta"""    
    # Unpack y1
    y1: int = Y[0]
    # Expected value of z
    return theta / (theta + 2.0) * y1


def theta_star(z: float, Y: data_type) -> float:
    """The M step of the EM algorithm.  Compute optimal theta given current parameters and z"""
    # Unpack Y
    (y1, y2, y3, y4) = Y
    # Weights for theta and 1-theta
    alpha: float = z + y4
    beta: float = y2 + y3
    # The optimal value of theta
    return alpha / (alpha + beta)


def EM(Y: data_type, theta: float) -> Tuple[float, int]:
    """Perform the EM algorithm to estimate theta using maximum likelihood."""
    # Maximum number of iterations
    max_iters: int = 10000
    # Threshold change in theta
    tol: float = 1e-12
    # Initialize 
    theta_prev: float = theta
    # Perform EM
    for i in range(max_iters):
        # E-step
        z: float = z_bar(theta, Y)
        # M-step
        theta = theta_star(z, Y)
        # Change in theta this step
        if abs(theta - theta_prev) < tol:
            break
        theta_prev = theta
    # Return the optimal theta and the number of iterations
    return (theta, i)


# Observed data values in this problem
(y1, y2, y3, y4) = (125, 18, 20, 34)
# Assemble these into a 4-vector
Y: data_type = (y1, y2, y3, y4)
# Total number of data points
N: int = sum(Y)
# Initial guess - based on moment matching
theta: float = ((y1 + y4) - (y2 + y3)) / N
iters: int
# Run EM on these imputs
(theta, iters) = EM(Y, theta)
# Report results
print(f'EM algorithm converged after {iters} iterations.')
print(f'Maximum likelihood estimate: theta = {theta:0.6f}.')