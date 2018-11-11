"""
Harvard Applied Math 207
Homework 9
Problem 1

Michael S. Emanuel
Sat Nov 10 15:08:13 2018
"""

# Let X be a random variable taking values in  ℝ2 . That is, X is a 2-dimensional vector. 
# Suppose that X is normally distributed as follows:
# X ~ N(mu, Sigma)
# where mu = [1; 2] and Sigma = [[4, 1.2];[1.2, 4]]
# That is, the pdf of X is 
# f_X(x) = 1 / 2pi sqrt(det(S)) exp{-1/2 (x-mu)T Sigma^-1 (x-mu)}
# In the following questions, we will denote the random variable corresponding to the 
# first component of X by X1 and the second component by X2 .

import numpy as np
from numpy import sqrt
from scipy.stats import norm
import matplotlib as mpl
import matplotlib.pyplot as plt

# Set default font size to 20
mpl.rcParams.update({'font.size': 25})

# *************************************************************************************************
# Set parameters in this problem
mu = np.array([1, 2])
sigma = np.array([[4, 1.2],[1.2, 4]])
mu1 = mu[0]
mu2 = mu[1]
sigma1 = sqrt(sigma[0,0])
sigma2 = sqrt(sigma[1,1])
rho = sigma[0,1] / (sigma1 * sigma2)

# Check that sigma is recovered properly
sigma_rec = np.array([[sigma1**2, rho*sigma1*sigma2],[rho*sigma1*sigma2, sigma2**2]])
assert np.array_equal(sigma, sigma_rec)

# *************************************************************************************************
# 1.1. Write down the two conditional distributions  fX1|X2,fX2|X1 

# Distribution of X1 conditional on X2 = x2 is:
# X1 ~ Normal(mu=mu_1 + sigma_1 / sigma_2 * rho * (x2 - mu_2), (1-rho^2)*sigma_1^2
# X2 ~ Normal(mu=mu_2 + sigma_2 / sigma_1 * rho * (x1 - mu_1), (1-rho^2)*sigma_2^2

# *************************************************************************************************
# 1.2. Write a Gibbs sampler for this distribution by sampling sequentially from 
# the two conditional distributions  fX1|X2, fX2|X1 .

def x1_cond_x2(x2: float, mu1, mu2, sigma1, sigma2, rho) -> float:
    """Draw samples of x1 conditional on x2"""
    # Conditional expectation of X1 given X2 = x2
    cond_mean = mu1 + (sigma1 / sigma2) * rho * (x2 - mu2)
    # Conditional variance doesn't depend on the value x2, just the variances and covarinces
    # cond_var = (1.0 - rho**2) * sigma_1**2
    cond_std = sqrt(1.0 - rho**2) * sigma1
    # Draw a random sample
    return np.random.normal(loc=cond_mean, scale=cond_std)


def x2_cond_x1(x1: float, mu1, mu2, sigma1, sigma2, rho) -> float:
    """Draw samples of x2 conditional on x1"""
    # Conditional expectation of X1 given X2 = x2
    cond_mean = mu2 + (sigma2 / sigma1) * rho * (x1 - mu1)
    # Conditional variance doesn't depend on the value x2, just the variances and covarinces
    # cond_var = (1.0 - rho**2) * sigma_1**2
    cond_std = sqrt(1.0 - rho**2) * sigma1
    # Draw a random sample
    return np.random.normal(loc=cond_mean, scale=cond_std)


def gibbs_sampler(x1_cond_x2, x2_cond_x1, N: int, init: np.ndarray):
    """Gibbs sample for the bivariate normal"""
    # Initialize x1 and x2
    x1, x2 = init
    # Preallocate storage for the samples
    samples = np.zeros((N+1, 2))

    # Bind the arguments mu1, mu2, sigma1, sigma2, rho for legibility
    x1_sampler = lambda x2: x1_cond_x2(x2, mu1, mu2, sigma1, sigma2, rho)
    x2_sampler = lambda x1: x2_cond_x1(x1, mu1, mu2, sigma1, sigma2, rho)
    
    # Initial sample
    samples[0, :] = init
    # Iterate N times
    for i in range(1, N, 2):
        # Sample one point, drawing x1 given x2
        x1 = x1_sampler(x2)
        samples[i+0, :] = (x1, x2)
        # Sample one point, drawing x2 given x1
        x2 = x2_sampler(x1)
        samples[i+1, :] = (x1, x2)        
    # The array samples is now fully populated; return it
    return samples


# *************************************************************************************************
# 1.3. Choose a thinning parameter, burn-in factor and total number of iterations that 
# allow you to take 10000 non-autocorrelated draws.


def draw_samples(gibbs_sampler, N: int, burnin: int, thinning: int):
    """Draw a set of samples from the Gibbs sampler with the specified burn-in and thinning parameters."""
    # Bind global variables
    global mu1, mu2
    # Set starting point to the mean
    init = np.array([mu1, mu2])
    # Compute number of raw samples required
    N_raw: int = burnin + N * thinning
    # Draw this many samples
    samples_raw = gibbs_sampler(x1_cond_x2, x2_cond_x1, N_raw, init)
    # Extract the ostensibly high quality samples
    return samples_raw[burnin::thinning]

# Desired number of samples
N: int = 10000
# Sett burn-in and thinning
burnin: int = 1000
thinning: int = 4
# Draw the samples with these parameters
# if 'samples' not in globals():
samples = draw_samples(gibbs_sampler, N, burnin, thinning)


# *************************************************************************************************
# 1.4. Plot a 2-d histogram of your samples, as well histograms of the X1 and X2  marginals. 
# Overlay on your histograms of the marginals a plot of the appropriate marginal density fitted 
# with parameters derived from your marginal samples.


def plot_2d(samples):
    """Plot the 2D histogram of samples"""
    # Global variables
    global mu1, mu2, sigma
    fig, ax = plt.subplots()
    fig.set_size_inches([16, 8])
    ax.set_title('Bivariate Normal Samples from Gibbs Sampler')
    ax.set_aspect('equal')
    x_plot = samples[:,0]
    y_plot = samples[:,1]
    # Plot the samples
    ax.scatter(x_plot, y_plot, color='b', s=1, label='samples')
    # Plot the mean
    ax.plot(mu1, mu2, color='r', marker='o', markersize=15, label='mean')
    # Compute the major and minor axis of the ellipse
    eig_vals, eig_vecs = np.linalg.eig(sigma)
    pca1 = sqrt(eig_vals[0]) * eig_vecs[:,0]
    pca2 = sqrt(eig_vals[1]) * eig_vecs[:,1]
    axis_multipliers = np.linspace(-2, 2, 2)
    maj_axis_x = mu1 + pca1[0] * axis_multipliers
    maj_axis_y = mu2 + pca1[1] * axis_multipliers
    min_axis_x = mu1 + pca2[0] * axis_multipliers
    min_axis_y = mu2 + pca2[1] * axis_multipliers
    # Plot the major and minor axes of the ellipse
    ax.plot(maj_axis_x, maj_axis_y, color='k', linewidth=4, label='major axis')
    ax.plot(min_axis_x, min_axis_y, color='k', linewidth=4, label='minor axis')
    ax.grid()

# Plot the 2D samples
# plot_2d(samples)

# Densities to overlay on the histogram plots
def density_x1(x1: float):
    """The theoretical marginal density of x1"""
    # Bind global variables
    global mu1, sigma1
    # Marginal density of x1 is normal with mean mu1 and standard deviation sigma1
    return norm.pdf(x1, loc=mu1, scale=sigma1)


def density_x2(x2: float):
    """The theoretical marginal density of x2"""
    # Bind global variables
    global mu2, sigma2
    # Marginal density of x1 is normal with mean mu1 and standard deviation sigma1
    return norm.pdf(x2, loc=mu2, scale=sigma2)


def plot_hist(samples, density, var_name: str):
    """Plot a histogram of samples of one variable overlaid with its density"""
    # Get min and max
    x_min = np.min(samples)
    x_max = np.max(samples)
    # evenly spaced x grid for density
    x_grid = np.linspace(x_min, x_max, 101)

    # Plot the samples and density
    fig, ax = plt.subplots()
    fig.set_size_inches([16, 8])
    ax.set_title(f'Marginal Samples of {var_name}')
    ax.hist(samples, bins=100, density=True, color='b', label='MCMC', )
    ax.plot(x_grid, density(x_grid), color='r', label='True', linewidth=5)
    ax.legend()
    ax.grid()

# Extract marginal samples of x1 and x2
marginal_x1 = samples[:, 0]
marginal_x2 = samples[:, 1]

# Plot the histogram of x1 and x2
plot_hist(marginal_x1, density_x1, 'x1')
plot_hist(marginal_x2, density_x2, 'x2')


# *************************************************************************************************
# 1.4. Present traceplots and autocorrelation plots for your marginal samples. 
# Is your choice of parameters justified? 

def make_traceplot(samples, var_name: str):
    """Make a traceplot for this variable"""
    fig, ax = plt.subplots()
    fig.set_size_inches([16, 8])
    ax.set_title(f'Trace Plot of {var_name}')
    ax.plot(samples, alpha=0.3, color='b')
    ax.grid()

# make_traceplot(marginal_x1, 'x1')
# make_traceplot(marginal_x2, 'x2')


def plot_autocorr(samples, var_name: str, maxlags: int = 50):
    """Plot the autocorrelation for this variable"""
    fig, ax = plt.subplots()
    fig.set_size_inches([16, 8])
    ax.set_title(f'Autocorrelation of {var_name}')    
    ax.acorr(samples - np.mean(samples), normed=True, maxlags=maxlags, color='b')
    ax.set_xlim([0, maxlags])
    ax.set_ylim([-0.1, 0.1])
    ax.grid()

plot_autocorr(marginal_x1, 'x1', 50)
plot_autocorr(marginal_x1, 'x1', 50)
