"""
Harvard IACS AM 207
Homework 5
Problem 1

Michael S. Emanuel
Tue Oct  9 18:27:22 2018
"""

import numpy as np
from numpy import sqrt, exp, pi
from scipy.integrate import cumtrapz
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from am207_utils import plot_style, arange_inc

# Set plot style
plot_style()

# *************************************************************************************************
# Question 1: We'll Always Have that Night Sampling in Monte Carlo
# *************************************************************************************************

# Let X be a random variable with distribution described by the following pdf:
# f_X(x) =
#  (1/12) * (x-1)    on (1,3]
# −(1/12)* (x−5)     on (3,5]
# (1/6)*(x-9)        on (7, 9]
# 0 otherwise

# Let h be the following function of XX:
# h(X)=(1/3 √2 π ) * exp{−(1/18)(X−5)^2}

# Compute 𝔼[h(X)]E[h(X)] via Monte Carlo simulation using the following sampling methods:

# *************************************************************************************************
# Compute  𝔼[h(X)]E[h(X)]  via Monte Carlo simulation using the following sampling methods:
# (1.1) Inverse Transform Sampling
# (1.2) Rejection Sampling with a uniform proposal distribution

# 1. Shared prerequisites for problem 1 (both parts)
def f_X(x: float) -> float:
    """The PDF for problem 1"""
    if x < 1:
        return 0
    elif x <= 3:
        return (1/12) * (x - 1)
    elif x <= 5:
        return (1/12) * (5 -x)
    elif x <= 7:
        return (1/6) * (x-5)
    elif x <= 9:
        return (1/6) * (9-x)
    else:
        return 0


def h(x: float) -> float:
    """The function h(X) whose expectation we want to take"""
    # The normalizing constant
    a: float = 1.0 / (3 * sqrt(2) * pi)
    # The term in the exponential
    u: float = -(1/18)*(x-5)*(x-5)
    return a * exp(u)


def expectation_mc(h, x_sample):
    """Take the expectation of a function given samples
    h:          The function whose expectation we want to take
    x_sample:   The samples of the function
    """
    # Evaluate h(x) on the samples and return the mean
    return np.mean(h(x_sample))


# *************************************************************************************************
def F_X(x: float) -> float:
    """Analytical CDF for problem 1"""
    x2 = x*x
    if x < 0:
        return 0
    elif x <= 3:
        return (1/12) * (x2 / 2 -x + 1/2)
    elif x <= 5:
        return F_X(3) + 1/12 * (5*x - x2/2 - 21/2)
    elif x <= 7:
        return F_X(5) + 1/6*(x2/2 - 5*x + 25/2)
    elif x <= 9:
        return F_X(7) + 1/6*(9*x - x2/2 - 77/2)
    else:
        return 1
        

# *************************************************************************************************
# 1. Visualize PDF and CDF

# Values for plots
plot_x = arange_inc(1, 9, 0.05)
plot_f = np.array([f_X(x) for x in plot_x])
plot_F = np.array([F_X(x) for x in plot_x])

# Plot the PDF f_X(x)
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('PDF $f_X(x)$')
ax.set_xlabel('x')
ax.set_ylabel('$f_X(x)$')
ax.set_xlim([1,9])
ax.plot(plot_x, plot_f, label='PDF')
ax.grid()
plt.show()

# Plot the CDF F_X(x)
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('CDF $F_X(x)$')
ax.set_xlabel('x')
ax.set_ylabel('$F_X(x)$')
ax.set_xlim([1,9])
ax.set_ylim([0,1])
ax.plot(plot_x, plot_F, label='CDF')
ax.legend()
ax.grid()
plt.show()


# *************************************************************************************************
# 1.1. Inverse Transform Sampling
def cdf_and_inverse(f, a, b, dx):
    """Generate a numerical inverse CDF to the PDF given by f(x)
    f:  The probability density function whose CDF is to be numerically inverted
    a:  The start of the support for f(x)
    b:  The end of the support for f(x)
    dx: The step size to use in sampling on [a, b]    
    """
    # Sample f_X(x) on the interval [a, b] with step size dx
    dx = 0.01
    sample_x = arange_inc(a, b, dx) 
    sample_f = np.array([f(x) for x in sample_x])
    
    # Numerical integral of F using cumtrapz library function
    sample_F = np.zeros_like(sample_f)
    sample_F[1:] = cumtrapz(sample_f, sample_x)
    # Normalize this to guarantee it ranges from [0, 1] notwithstanding any round-off
    sample_F = sample_F / sample_F[-1]
    
    # Use the Pchip interpolator b/c this guarantees that monotonic input is sent to monotonic output
    # Numerical CDF using interpolation
    F = PchipInterpolator(sample_x, sample_F)
    # Numerical inverse CDF using interpolation
    # Silence these warnings; it's OK, the splined inverse interpolant is horizontal in places but it works
    with np.errstate(divide='ignore', invalid='ignore'):
        F_inv = PchipInterpolator(sample_F, sample_x)
    # Return the splined CDF and inverse CDF function
    return F, F_inv

# Get the CDF and inverse CDF for the given f
F_X, F_X_inv = cdf_and_inverse(f_X, 1.0, 9.0, 0.01)


# *************************************************************************************************
# 1.1 Take samples and compute expectation
def samples_x_inv_trans(F_X_inv, size):
    """
    Sample random variable X using Inverse Transform Sampling
    F_X_inv: inverse of the CDF
    size:    size of the array to generate
    """
    # Sample u uniformly on [0, 1]
    u = np.random.uniform(size=size)
    # Apply the inverse transform
    return F_X_inv(u)

# Generate 1,000,000 samples for x
sample_size: int = 10**6
x_samples_its = samples_x_inv_trans(F_X_inv, size=sample_size)
# Compute E_f[H] on these samples
exp_h_its: float = expectation_mc(h, x_samples_its)
# Report the results
print(f'Expectation of h(x) using Inverse Transform Sampling: {exp_h_its:0.6f}')

# *************************************************************************************************
# 1.2. Rejection Sampling with a uniform proposal distribution 
# (rejection sampling in a rectangular box with uniform probability of sampling any x)

def samples_x_reject(f_X, size, a: float, b: float, y_max: float):
    """Sample random variable X using Rejection Sampling with a uniform proposal distribution
    f_X:   probability density function for f(x)
    size:  size of the array to generate
    a:     start of the support of f(x); left side of rectangular box
    b:     end of the support of f(x); right side of rectangular box
    y_max: maximum of f(x) on [a, b]; height of rectangular box
    """
    # Preallocate space for the drawn samples
    x_samples = np.zeros(size)
    # Count the number of samples drawn and attempts
    idx: int = 0
    attempts: int = 0
    # Continue drawing new samples until we've collected size of them
    while idx < size:
        # Draw a random value of x on [a, b]
        x = np.random.uniform(a, b)
        # Draw a random value of y on [0, y_max]; if y <= f_X(x), keep this sample
        if np.random.uniform(0, y_max) <= f_X(x):
            # Save the sample in slot idx, then increment idx
            x_samples[idx] = x
            idx += 1
        # Always increment attempts
        attempts += 1
    # Return the list of samples as well as the number of attempts
    return x_samples, attempts

# Generate samples with rejection sampling
# Maximum value of f(X) occurs at x=7
y_max: float = f_X(7)
x_samples_rs, attempts = samples_x_reject(f_X, sample_size, 1.0, 9.0, y_max)
# Report number of trials
print(f'Drew {sample_size} samples for x ~ f(x) in {attempts} attempts.')

# Compute E_f[H] on these samples
exp_h_rs: float = expectation_mc(h, x_samples_rs)
# Report the results
print(f'Expectation of h(x) using Rejection Sampling: {exp_h_rs:0.6f}')
