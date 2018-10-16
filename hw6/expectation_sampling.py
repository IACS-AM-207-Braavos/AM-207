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
# import matplotlib.mlab as mlab
# from matplotlib import cm
# import pandas as pd

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
print(f'The exact answer with scipy.integrate.quad is {exp_exact:0.8f} to a tolerance of {tol_exact:0.3e}.')

# *************************************************************************************************
# 1.1. Rejection sampling with a normal proposal distribution and appropriately chosen parameters 
# Visualize PDF and design a proposal distribution

# Values for plots
plot_x = arange_inc(1, 9, 0.05)
plot_f = f(plot_x)

# Create a proposal distribution by hand by looking at the chart
# Experiment to get M as small as possible
mu1 = 5.8
sigma1: float = np.std(plot_x)*0.9
# Proposal distribution g(x) (NOT majorized)
g1: funcType = lambda x : norm.pdf(x, loc=mu1, scale=sigma1)
plot_g1 = g1(plot_x)
M1: float = np.max(plot_f / plot_g1)*1.01
plot_g1_maj = M1 * plot_g1
print(f'Proposal Distribution and Majorizer for rejection sampling.')
print(f'mu={mu1:0.6f}, sigma={sigma1:0.6f}, M={M1:0.6f}')
# Define the sampling distribution for the chosen proposal distribution g(x)
g1_sample = lambda : norm.rvs(loc=mu1, scale=sigma1)

# Plot the PDF f_X(x) and the majorizing distribution Mg(x)
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('PDF $f_X(x)$ and its Majorizer $Mg(x)$')
ax.set_xlabel('x')
ax.set_ylabel('$f_X(x)$')
ax.set_xlim([1,9])
ax.plot(plot_x, plot_f, label='PDF')
ax.plot(plot_x, plot_g1_maj, label='Mg(x)')
ax.legend()
ax.grid()
plt.show()

 
# *************************************************************************************************
# 1.1 Compute E_f[h] using rejection sampling from a proposal distribution
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
    x_samples_rs, attempts = rejection_sample_proposal(f, g1, g1_sample, M1, sample_size)
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

# Define uniform proposal distribution's density (will only be used on [a, b]
def g_unif(x):
    if isinstance(x, np.ndarray):
        y = np.zeros_like(x)
        mask = (a <= x) & (x <= b)
        y[mask] = 1.0 / (b-a)
    if isinstance(x, float) or isinstance(x, int):
        if a <= x and x <= b:
            return 1.0 / (b-a)
        else:
            return 0


# Draw samples uniformly on [a, b]; we keep them all because we're doing importance sampling
x_samples_unif = np.random.uniform(low=a, high=b, size=sample_size)
print(f'Drew {sample_size} random samples on uniform proposal distribution.')
# The importance weights are just the ratio f(x) / g(x) on each sample
# Here the proposal g(x) is uniform, so g(x) = 1/(b-a)
wts_unif = f(x_samples_unif) * (b-a)
# Compute E_f[H] with these samples and importance weights; report results
exp_h_unif = expectation_mc(h, x_samples_unif, wts_unif)
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
g2: funcType = lambda x : norm.pdf(x, loc=mu2, scale=sigma2)
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
print(f'Expectation of h(x) using Normal Proposal Distribution & Importance Weights: {exp_h_norm:0.6f}; '
      f'Error {err_norm:0.2e}')


# *************************************************************************************************
# 1.4. So far (in HWs 5 and 6) we've computed estimates of  𝔼[h(X)]  for the following list of methods:

# (a) Inverse Transform Sampling
# (b) Rejection Sampling with a uniform proposal distribution 
#     (rejection sampling in a rectangular box with uniform probability of sampling any x)
# (c) Rejection sampling with a normal proposal distribution and appropriately chosen parameters (aka rejection on steroids)
# (d) Importance sampling with a uniform proposal distribution
# (e) Importance sampling with a normal proposal distribution and appropriately chosen parameters.

# Compute the variance of each estimate of  𝔼[h(X)] you calculated in this list. 
# Which sampling methods and associated proposal distributions would you expect based on discussions from lecture 
# to have resulted in lower variances? How well do your results align with these expectations?

# Lecture 10, p. 18: 
# For "regular" sampling, V_hat = V_f[h(x)] / N
# For importance sampling, V_hat = V_g[w(x) h(x)] / N

def weighted_variance(x: np.ndarray, wts: np.ndarray):
    """Compute the variance of an array with weights"""
    # Compute the weighted average
    wtd_avg = np.average(a=x, weights=wts)
    # Compute the difference to the weighted average
    dx = x - wtd_avg
    # The weighted variance is the weighted average of dx*dx    
    return np.average(a=dx*dx, weights=wts)

def exp_mc_var(h: funcType, x_samples):
    """Esimate variance of h(x) on sampled x."""
    # Number of samples
    N: int = x_samples.shape[0]
    # Compute the expected variance of h(x) with probability f(x): V_f[h(x)]
    h_x: np.ndarray = h(x_samples)
    f_x: np.ndarray = f(x_samples)
    h_var: float = weighted_variance(h_x, f_x)
    # Return the sample variance over N
    return h_var / N


def exp_mc_var_wtd(h: funcType, g: funcType, x_samples, wts):
    """Esimate variance of h(x) on sampled x."""
    # Number of samples
    N: int = x_samples.shape[0]
    # Compute the expected variance of w(x)h(x) with probability g(x): V_g[w(x)h(x)]
    wh_x: np.ndarray = wts * h(x_samples)
    g_x: np.ndarray = g(x_samples)
    wh_var: float = weighted_variance(wh_x, g_x)
    # Return the sample variance over N
    return wh_var / N


# *************************************************************************************************
print(f'\nExpected variance on MC integral with 5 sampling methods and {sample_size} samples.')

# The expected variance and standard deviation for all three sampling based approaches.
exp_var_samp = exp_mc_var(h, x_samples_norm)
exp_std_samp = sqrt(exp_var_samp)
# Report results
print(f'Estimated variance for all three sampling estimates with {sample_size} samples:')
print(f'Estimated variance = {exp_var_samp:0.3e}, standard deviation = {exp_std_samp:0.3e}.')

# Exact answer
var_func = lambda x : (h(x)-exp_exact)**2
var_func_f = lambda x : var_func(x) * f(x)
exp_var_exact = quad(var_func_f, a, b)[0] / sample_size
exp_std_exact = sqrt(exp_var_exact)
print(f'Estimated variance with integrate.quad = {exp_var_exact:0.3e}, standard deviation = {exp_std_exact:0.3e}')

# (d) Importance Sampling with unifrom proposal distribution
var_imp_unif = exp_mc_var_wtd(h, g_unif, x_samples_unif, wts_unif)
std_imp_unif = sqrt(var_imp_unif)
print(f'Importance Weighting with Uniform Proposal: variance = {var_imp_unif:0.3e}, std = {std_imp_unif:0.3e}.')

# (e) Importance Sampling with normal proposal distribution
var_imp_norm = exp_mc_var_wtd(h, g2, x_samples_norm, wts_norm)
std_imp_norm = sqrt(var_imp_unif)
print(f'Importance Weighting with Normal Proposal: variance = {var_imp_norm:0.3e}, std = {std_imp_norm:0.3e}.')
