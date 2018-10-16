"""
Harvard IACS AM 207
Homework 6
Problem 1

Michael S. Emanuel
Mon Oct 15 17:13:59 2018
"""

import numpy as np
from numpy import sqrt, exp, pi
# import scipy.stats
# import scipy.special
from scipy.stats import norm
from scipy.integrate import quad
from time import time

# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import cm
import pandas as pd

from am207_utils import plot_style, arange_inc, funcType

# Set plot style
plot_style()

# *************************************************************************************************
# Question 1: Can I sample from F-R-I-E-N-D-S without rejection? It's Important!
# *************************************************************************************************

# In HW 5 we were introduced to X, a random variable with distribution described by the following pdf:
# f_X(x) =
#  (1/12) * (x-1)    on (1,3]
# −(1/12)* (x−5)     on (3,5]
# (1/6)*(x-9)        on (7, 9]
# 0 otherwise

# We were also introduced to  hh  the following function of X :
# h(X)=(1/3 √2 π ) * exp{−(1/18)(X−5)^2}

# Compute  𝔼[h(X)] via Monte Carlo simulation using the following sampling methods

# *************************************************************************************************
# 1. Shared prerequisites for problem 1 (both parts)
def f_scalar(x: float):
    """The PDF, defined on scalars"""
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
    

def f_vec(x: np.ndarray) -> np.ndarray:
    """The PDF for problem 1; vectorized implementation"""
    # Initialize return vector of zeros
    y = np.zeros_like(x)
    # x in [1, 3)
    mask = (1 <= x) & (x < 3)
    y[mask] = (1/12) * (x[mask] - 1)
    # x in [3, 5)
    mask = (3 <= x) & (x < 5)
    y[mask] = (1/12) * (5 - x[mask])
    # x in [5, 7)
    mask = (5 <= x) & (x < 7)
    y[mask] = (1/6) * (x[mask]-5)
    # x <= 9
    mask = (7 < x) & (x <= 9)
    y[mask]= (1/6) * (9-x[mask])
    # Return the array of ys
    return y


def f(x):
    """The PDF; handles sclars or vectors appropriately"""
    if isinstance(x, np.ndarray):
        return f_vec(x)
    if isinstance(x, float) or isinstance(x, int):
        return f_scalar(x)
    raise ValueError('f(x) supports np.ndaraay, float, int.')


def h(x: float) -> float:
    """The function h(X) whose expectation we want to take"""
    # The normalizing constant
    a: float = 1.0 / (3 * sqrt(2) * pi)
    # The term in the exponential
    u: float = -(1/18)*(x-5)*(x-5)
    return a * exp(u)

# Limits of the support of f(x)
a: float = 1.0
b: float = 9.0
# Calculate the exact answer using scipy.integrate.quad
fh = lambda x : f(x) * h(x)
exp_exact, tol_exact = quad(fh, a, b)
print(f'The exact answer with scipy.integrate.quad is {exp_exact} to a tolerance of {tol_exact:0.3e}.')

# *************************************************************************************************
# 1.1. Rejection sampling with a normal proposal distribution and appropriately chosen parameters 
# Visualize PDF and design a proposal distribution

# Values for plots
plot_x = arange_inc(1, 9, 0.05)
plot_f = f(plot_x)

# Create a proposal distribution by hand by looking at the chart
# Experiment to get M as small as possible
mu = 5.8
sigma: float = np.std(plot_x)*0.9
# Proposal distribution g(x) (NOT majorized)
g: funcType = lambda x : norm.pdf(x, loc=mu, scale=sigma)
plot_g = g(plot_x)
M: float = np.max(plot_f / plot_g)*1.01
plot_g_maj = M * plot_g
print(f'mu={mu:0.6f}, sigma={sigma:0.6f}, M={M:0.6f}')
# Define the sampling distribution for the chosen proposal distribution g(x)
g_sample = lambda : norm.rvs(loc=mu, scale=sigma)

# Plot the PDF f_X(x) and the majorizing distribution Mg(x)
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('PDF $f_X(x)$ and its Majorizer $Mg(x)$')
ax.set_xlabel('x')
ax.set_ylabel('$f_X(x)$')
ax.set_xlim([1,9])
ax.plot(plot_x, plot_f, label='PDF')
ax.plot(plot_x, plot_g_maj, label='Mg(x)')
ax.legend()
ax.grid()
plt.show()

 
# *************************************************************************************************
def rejection_sample_proposal(f: funcType, g: funcType, g_sample: funcType, M: float, size: int):
    """
    Perform rejection sampling: 
    f: the probability density (PDF) we wish to sample
    g: the proposal distribution we use for candidate sample points
    g_sample: function to used to draw random samples with PDF g(x)
    M: the majorizer such that f(x) < Mg(x) on the support of f
    size: the number of samples
    """
    
    # Preallocate space for the drawn samples
    x_samples = np.zeros(size)
    # Count the number of samples drawn and attempts
    idx: int = 0
    attempts: int = 0
    # Continue drawing new samples until we've collected size of them
    # Folow the recipe in Lecutre 10, p. 14
    while idx < size:
        # Draw a random value of x from the proposal distribution
        x = g_sample()
        # Draw a random value of y on [0, 1]
        y: float = np.random.uniform()
        if y * M * g(x) <= f(x):
            # Save the sample in slot idx, then increment idx
            x_samples[idx] = x
            idx += 1
        # Always increment attempts
        attempts += 1
    # Return the list of samples as well as the number of attempts
    return x_samples, attempts

def expectation_mc(h, x_sample, wts = None):
    """Take the expectation of a function given samples
    h:          The function whose expectation we want to take
    x_sample:   The samples of the function
    weights:    The importance weights to use; optional, defaults to None (ignored)
    """
    E: float
    if wts is None:
        # Evaluate h(x) on the samples and return the mean if there were no weights
        E = np.mean(h(x_sample))
    else:
        # Evaluate h(x) on the samples and return the weighted average with the given importance weights
        E = np.average(a=h(x_sample), weights=wts)
    return E

# Set a consistent sample size used in all three parts of problem 1
sample_size: int = 10**4

# Draw a sample of x's from the normal proposal distribution
if 'x_samples_rs' not in globals():
    t0 = time()
    x_samples_rs, attempts = rejection_sample_proposal(f, g, g_sample, M, sample_size)
    t1 = time()
    elapsed = t1 - t0
    print(f'Generated {sample_size} samples in {attempts} attempts using rejection sampling; {elapsed:0.2f} seconds.')
else:
    print(f'Using {sample_size} samples previously generated by rejection sampling.')

# Compute E_f[H] on these samples and report results
exp_h_rs: float = expectation_mc(h, x_samples_rs)
err_rs: float = np.abs(exp_h_rs - exp_exact)
print(f'Expectation of h(x) using Rejection Sampling: {exp_h_rs:0.6f}.  '
      f'Error = {err_rs:0.2e}\n')


# *************************************************************************************************
# 1.2. Importance sampling with a uniform proposal distribution

# Draw samples uniformly on [a, b]; we keep them because we're doing importance sampling
x_samples_unif = np.random.uniform(low=a, high=b, size=sample_size)
print(f'Drew {sample_size} random samples on uniform proposal distribution.')
# The importance weights are just the ratio f(x) / g(x) on each sample
# Here the proposal g(x) is uniform, so g(x) = 1/(b-a)
wts_un = f(x_samples_unif) * (b-a)
# Compute E_f[H] with these samples and importance weights; report results
exp_h_unif = expectation_mc(h, x_samples_unif, wts_un)
err_unif: float = np.abs(exp_h_unif - exp_exact)
print(f'Expectation of h(x) using Uniform Sampling and Importance Weights: {exp_h_unif:0.6f}; '
      f'Error = {err_unif:0.2e}\n')

# *************************************************************************************************
# 1.3. Importance sampling with a normal proposal distribution and appropriately chosen parameters

# Create a proposal distribution by hand by looking at the chart
# This plot is to match f(x) h(x), not f(x)
plot_h = h(plot_x)
plot_fh = plot_f * plot_h
# Experiment to get M as small as possible
mu2 = 5.8
sigma2: float = np.std(plot_x)*0.9
# Proposal distribution g(x) (NOT majorized)
g2: funcType = lambda x : norm.pdf(x, loc=mu, scale=sigma)
plot_g2 = g2(plot_x)
M2: float = np.max(plot_fh / plot_g2)*1.01
plot_g2_maj = M2 * plot_g2
print(f'mu={mu2:0.6f}, sigma={sigma2:0.6f}, M2={M2:0.6f}')
# Define the sampling distribution for the chosen proposal distribution g(x)
# g2_sample = lambda size: norm.rvs(loc=mu2, scale=sigma2, size=size)

# Plot the PDF f_X(x) and the majorizing distribution Mg(x)
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('$f_X(x) h(x)$ and its Majorizer $Mg(x)$')
ax.set_xlabel('x')
ax.set_ylabel('$f_X(x) h(x)$')
ax.set_xlim([1,9])
ax.plot(plot_x, plot_fh, label='$f_x(x) h(x)$')
ax.plot(plot_x, plot_g2_maj, label='$Mg(x)$')
ax.legend()
ax.grid()
plt.show()

# Draw a sample of x's from the normal proposal distribution; keep them all because we're importance sampling
x_samples_norm = norm.rvs(loc=mu2, scale=sigma2, size=sample_size)
# The importance weights w(x) = f(x) / g(x)
wts_norm = f(x_samples_norm) / g2(x_samples_norm)
# Compute the E_f[H] on these samples and importance weights; report results
exp_h_norm = expectation_mc(h, x_samples_norm, wts_norm)
err_norm = np.abs(exp_h_norm - exp_exact)
print(f'Expectation of h(x) using Normal Proposal Distribuion & Importance Weights: {exp_h_norm:0.6f};'
      f'Error {err_norm:0.2e}')


# *************************************************************************************************
# 1.4. So far (in HWs 5 and 6) we've computed estimates of  𝔼[h(X)]  for the following list of methods:

# (a) Inverse Transform Sampling
# (b) Rejection Sampling with a uniform proposal distribution (rejection sampling in a rectangular box 
# (c) with uniform probability of sampling any x)
# (d) Rejection sampling with a normal proposal distribution and appropriately chosen parameters (aka rejection on steroids)
# (e) Importance sampling with a uniform proposal distribution
# (e) Importance sampling with a normal proposal distribution and appropriately chosen parameters.