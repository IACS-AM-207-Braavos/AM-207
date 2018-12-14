"""
Michael S. Emanuel
Thu Dec 13 16:05:11 2018
"""

# core
import numpy as np
from numpy import log, exp
import scipy.stats
# probability
import pymc3 as pm
# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import arviz as az
# from arviz import plot_trace, plot_autocorr
# misc
import tqdm
from am207_utils import load_vartbl, save_vartbl, arange_inc
from typing import Dict

# *************************************************************************************************
# Q3: Exploring Temperature in Sampling and Optimiztion
# *************************************************************************************************
# At various times in class we've discussed in very vague terms the relation between 
# "temperature" and sampling from or finding optima of distributions. 
# Promises would invariably be made that at some later point we'd discuss the concept of temperature 
# and sampling/optima finding in more detail. 
# Let's take this problem as an opportunity to keep our promise.

# Let's start by considering the function 𝑓(𝑥,𝑦) defined in the following code cell. 
# 𝑓(𝑥,𝑦) is a mixture of three well separated Gaussian probability densities.

# *************************************************************************************************
# Load persisted table of variables
fname: str = 'temperature.pickle'
vartbl: Dict = load_vartbl(fname)

# Set plot style
mpl.rcParams.update({'font.size': 20})

# *************************************************************************************************
# This code provided as part of the problem statement
# make_cov assembles a 2x2 rotation matrix for angle theta
make_cov = lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
# Three angles chosen for rotations
theta_vec = (5.847707364986893, 5.696776968254305, 1.908095937315489)
theta1, theta2, theta3 = theta_vec

# define gaussian mixture 1 
cov1 = make_cov(theta1)
sigma1 = np.array([[2, 0],[0, 1]])
mvn1 = scipy.stats.multivariate_normal([12, 7], cov=cov1 @ sigma1 @ cov1.T)

# define gaussian mixture 2
cov2 = make_cov(theta2)
sigma2 = np.array([[1, 0],[0, 3]])
mvn2 = scipy.stats.multivariate_normal([-1, 6], cov=cov2 @ sigma2 @ cov2.T)

# define gaussian mixture 3
cov3 = make_cov(theta3)
sigma3 = np.array([[.4, 0],[0, 1.3]])
mvn3 = scipy.stats.multivariate_normal([3,-2], cov=cov3 @ sigma3 @ cov3.T)

# Define f(X) as an unnormalized mixture of multivariate normals
f = lambda xvec: mvn1.pdf(xvec) + mvn2.pdf(xvec) + .5*mvn3.pdf(xvec)
# Define p(x, y) as an unnormalized probability density function (pdf)
p = lambda x, y: f([x,y])

# *************************************************************************************************
# Part A Visualization and Metropolis
# *************************************************************************************************

# *************************************************************************************************
# A1. Visualize  𝑝(𝑥,𝑦)  with a contour or surface plot. 
# Make sure to title your plot and label all axes. 
# What do you notice about  𝑝(𝑥,𝑦) ? Do you think it will be an easy function to sample?

# Range of x and y
x_min: float = -8.0
x_max: float = 18.0
y_min: float = -6.0
y_max: float = 14.0

# Step size for x and y
dx: float = 0.125
dy: float = 0.125
# Discrete sample points for x and y
xx = arange_inc(x_min, x_max, dx)
yy = arange_inc(y_min, y_max, dy)
# Grid of x and y
grid_size_x: int = len(xx)
grid_size_y: int = len(yy)
# Grid of sample points for contour plots
x_grid, y_grid = np.meshgrid(xx, yy)

# Grid of p(x, y)
try:
    # raise ValueError
    p_grid = vartbl['p_grid']
    print(f'Loaded grid of p(x, y))')
except:    
    p_grid = np.zeros((grid_size_y, grid_size_x))
    for j, x in enumerate(xx):
        for i, y in enumerate(yy):
            p_grid[i, j] = p(x, y)
    # Approximate normalization constant
    norm: float = np.sum(p_grid)*dx*dy
    p_grid = p_grid / norm
    # Save normalized grid
    vartbl['p_grid'] = p_grid
    save_vartbl(vartbl, fname)


def plot_contour(x_grid, y_grid, p_grid, title):
    # Get min and max from grids
    x_min: float = np.min(x_grid[0,:])
    x_max: float = np.max(x_grid[0,:])
    y_min: float = np.min(y_grid[:,0])
    y_max: float = np.max(y_grid[:,0])
    
    # Plot the contours of f(x,y)
    fig, ax = plt.subplots(figsize=[13, 10])
    ax.set_title(title)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(arange_inc(x_min, x_max, 2.0))
    ax.set_yticks(arange_inc(y_min, y_max, 2.0))
    ax.set_aspect('equal', 'datalim')
    # compute desired contour levels
    p_max = np.max(p_grid)
    num_levels = 8
    step_size = p_max / (num_levels-1)
    half_step = 0.5*step_size
    levels = arange_inc(half_step, p_max - half_step, step_size)
    # Generate contour plot
    cs = ax.contour(x_grid, y_grid, p_grid, levels=levels, linewidths=2)
    fig.colorbar(cs, ax=ax)
    ax.grid()
    return fig, ax

# Plot contours of p(x, y)
fig = plot_contour(x_grid, y_grid, p_grid, 'Contours of $p(x, y)$')

# *************************************************************************************************
# A2. Generate 20000 samples from  𝑝(𝑥,𝑦) using the Metropolis algorithm. 
# Pick individual gaussian proposals in 𝑥 and 𝑦 with  𝜎=1, initial values, 
# burnin parameters, and thinning parameter. 
# Plot traceplots of the  𝑥  and  𝑦  marginals as well as autocorrelation plots. 
# Plot a pathplot of your samples. 
# Based on your visualizations, has your Metropolis sampler generated 
# an appropriate representation of the distribution  𝑝(𝑥,𝑦) ?
# A pathplot is just your samples trace overlaid on your pdf, so that you can see how the sampler traversed. 

# mu, sigma, and weight for Gaussian 1
mu_1 = np.array([[12, 7]])
R_1 = make_cov(theta1)
cov_1 = R_1 @ sigma1 @ R_1.T
weight_1 = 1.0 / 2.5
#    
# mu, sigma, and weight for Gaussian 2
mu_2 = np.array([[-1, 6]])
R_2 = make_cov(theta2)
cov_2 = R_2 @ sigma2 @ R_2.T
weight_2 = 1.0 / 2.5

# mu, sigma, and weight for Gaussian 3
mu_3 = np.array([[3, -2]])
R_3 = make_cov(theta3)
cov_3 = R_3 @ sigma3 @ R_3.T
weight_3 = 0.5 / 2.5

# Assemble into one mixture distribution
weights = np.array([weight_1, weight_2, weight_3])
mus = np.array([mu_1, mu_2, mu_3])

# *************************************************************************************************
# Sample using Metropolis

# Citation: this sampler was presented in lab 9
def metropolis(logp, proposal, step_size, num_samples, X_init):
    """Metropolis sampler"""
    # Initialize array of samples
    samples=np.zeros((num_samples, K))
    # Initialize X_prev to the initial point that was passed in
    X_prev = X_init
    accepted = 0
    # Drap num_samples sample points
    for i in tqdm.tqdm(range(num_samples)):
        X_star = proposal(X_prev, step_size)
        logp_star = logp(X_star)
        logp_prev = logp(X_prev)
        logpdf_ratio = logp_star -logp_prev
        u = np.random.uniform()
        if np.log(u) <= logpdf_ratio:
            samples[i] = X_star
            X_prev = X_star
            accepted += 1
        # We always draw a sample; whether it was accepted just determines whether we take the step or not
        else:
            samples[i]= X_prev
    # Return the samples and the acceptance ratio
    return samples, accepted


# The log-probability function
def logp(X: np.array):
    """Log probability at this point"""
    # unpack array X into scalars (x, y)
    (x, y) = X
    return log(p(x, y))


# Crate a proposal distribution with sigma=1 as per problem statement
def proposal(X, step_size):
    """Normal proposal distribution; step_size is passed from metropolis"""
    return np.random.normal(loc=X, scale=step_size * np.ones(K))

# Number of samples
num_samples: int = 20000
# Number of dimensions
K: int = 2

# Set the starting point as the weighted average of the clusters
X_init = np.average(mus, axis=0, weights=weights).squeeze()
# Set the stepsize to 1
step_size = 1.0
# Set tuning and thinning
tuning: int = 10000
thinning: int = 10
# Compute total number of raw samples required to generate num_samples
num_samples_raw: int = tuning + thinning * num_samples

# Load the samples if available
try:
    # raise ValueError
    raw_samples = vartbl['raw_samples']
    samples = vartbl['samples']
    acceptance_ratio = vartbl['acceptance_ratio']
    print(f'Loaded metropolis samples')
except:
    # Draw samples
    print(f'Drawing {num_samples_raw} samples with Metropolis algorithm...')
    raw_samples, accepted = metropolis(logp, proposal, step_size, num_samples_raw, X_init)
    acceptance_ratio = accepted / num_samples_raw   
    # Thin the samples
    samples = raw_samples[tuning::thinning]
    # Save samples to vartbl
    vartbl['raw_samples'] = raw_samples
    vartbl['samples'] = samples
    vartbl['acceptance_ratio'] = acceptance_ratio
    save_vartbl(vartbl, fname)

# Report acceptance ratio
print(f'Acceptance ratio was {acceptance_ratio:0.4f}.')

# Make trace by hand
trace = dict()
trace['x'] = samples[:,0]
trace['y'] = samples[:,1]

def plot_trace(trace, title):
    """Generate a trace plot with Arviz"""
    plot_kwargs = {'color':'b'}
    axs = az.plot_trace(trace, trace_kwargs=plot_kwargs)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=16)
    # ax0, ax1 = axs[0,0], axs[0,1]
    return fig, axs


def plot_autocorr(trace, title):
    """Generate an autocorr plot with Arviz"""
    axs = az.plot_autocorr(trace, max_lag=20)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=16)
    # ax0, ax1 = axs[0,0], axs[0,1]
    return axs


def plot_path(samples, x_grid, y_grid, p_grid, title):
    """Generate a cloud of sample points"""
    # Extract x and y from the samples
    x = samples[:,0]
    y = samples[:,1]
    # Thin out x and y for the plot
    thin = 10
    x = x[::thin]
    y = y[::thin]
    # Start with the contour plot
    fig, ax = plot_contour(x_grid, y_grid, p_grid, title)
    ax.set_title(title)
    # Add path of sample points
    ax.plot(x, y, color='r', linewidth=1, marker='o', markersize=2.5, alpha=0.25)
    return fig


def plot_cloud(samples, x_grid, y_grid, p_grid, title):
    """Generate a cloud of sample points"""
    # Extract x and y frmo the samples
    x = samples[:,0]
    y = samples[:,1]
    # Thin out x and y for the plot
    thin = 1
    x = x[::thin]
    y = y[::thin]
    # Start with the contour plot
    fig, ax = plot_contour(x_grid, y_grid, p_grid, title)
    # Add path of sample points
    ax.plot(x, y, color='r', linewidth=0, marker='o', markersize=1, alpha=0.15)
    return fig

# Generate the traceplot
fig = plot_trace(trace, 'Trace Plot of x and y')
# Generate the autocorr plot
fig = plot_autocorr(trace, 'Autocorrelation Plot of x and y')
# Plot the path taken by the sampler
fig = plot_path(samples, x_grid, y_grid, p_grid, 'Path Plot for Metropolis Samples')

# *************************************************************************************************
# Part B: Changing pdfs using temperature
# Given a function 𝑝(𝑥)  we can rewrite that function in following way:

# 𝑝(𝑥)=𝑒−(−log(𝑝(𝑥))
# So if define the energy density for a function as  𝐸(𝑥)≡−log𝑝(𝑥) 
# We can now aim to sample from the function parameratized by a Temperature  𝑇 .

# 𝑝(𝑥|𝑇)=𝑒xp(−1/𝑇 𝐸(𝑥)) =𝑝(x) ^(1/𝑇)
 
# If we set T=1 we're sampling from our original function  𝑝(𝑥) .

# *************************************************************************************************
# B1 In line with A1, visualize modified pdfs (dont worry about normalization) 
# by setting the temperatures to  𝑇=10  and  𝑇=0.1 .

#    def E(X: np.ndarray):
#        """Energy function for this problem"""
#        return -logp(X)


def logp_temp(X: np.ndarray, T: float):
    """log probability function with a temperature parameter T"""
    return logp(X) / T


def p_temp(x: float, y: float, T: float):
    """Probability density including a temperature parameter T"""
    return np.power(p(x, y), 1.0/T)

# Draw contour grid at selected temperatures
try:
    # raise ValueError
    p_grid_by_temp = vartbl['p_grid_by_temp']
except:
    p_grid_by_temp = dict()

# List of temperatures to sample from
temps = np.array([0.1, 1.0, 3.0, 7.0, 10.0])
# Build a probability grid at each temperature if it's not in the table
for T in temps:
    if T not in p_grid_by_temp:
        p_grid_T = np.zeros((grid_size_y, grid_size_x))
        for j, x in enumerate(xx):
            for i, y in enumerate(yy):
                # The pdf at this point with the given temperature
                p_grid_T[i, j] = p_temp(x, y, T)
        # Approximate normalization constant
        norm: float = np.sum(p_grid_T)*dx*dy
        p_grid_T = p_grid_T / norm
        # Save this grid to the table
        fig = p_grid_by_temp[T] = p_grid_T
    else:
        # Load this grid from the table
        p_grid_T = p_grid_by_temp[T]

# Save vartbl with updated entries
vartbl['p_grid_by_temp'] = p_grid_by_temp
save_vartbl(vartbl, fname)

# Temperatures for contour plot
temps_contour = np.array([0.1, 10.0])
for T in temps_contour:
    # Build a contour plot at this temperature
    plot_contour(x_grid, y_grid, p_grid_T, f'Contours of $p(x, y)$ at temperature T={T}')


# *************************************************************************************************
# B2. Modify your Metropolis algorithm above to take a temperature parameter T 
# as well as to keep track of the number of rejected proposals.
# Generate 20000 samples from 𝑝(𝑥,𝑦) for each of the following temperatures: {0.1, 1, 3, 7, 10}. 
# Construct histograms of the marginals, traceplots, autocorrelation plots, 
# and a pathplot for your samples at each temperature. 
# What happens to the number of rejections as temperature increases? 
# In the limits  𝑇→0  and  𝑇→∞  what do you think your samplers will do?

# Draw samples by temperature
try:
    samples_by_temp = vartbl['samples_by_temp']
    acceptance_by_temp = vartbl['acceptance_by_temp']
except:
    samples_by_temp = dict()
    acceptance_by_temp = dict()

# Compute a sample for each temperature if it's not in the table
for T in temps:
    if T not in samples_by_temp:
        # create p(X;T) in situ by binding T
        logp_T = lambda X : logp_temp(X, T)
        # Draw samples
        print(f'Drawing {num_samples_raw} samples at temperature T={T}')
        raw_samples_T, accepted_T = metropolis(logp_T, proposal, step_size, num_samples_raw, X_init)
        # Save samples and acceptance ratio
        samples_by_temp[T] = raw_samples_T
        acceptance_by_temp[T] = accepted_T / num_samples_raw
    else:
        print(f'Loaded {num_samples_raw} samples at temprerature T={T}')
    # Report acceptance ratio
    print(f'Acceptance ratio at T={T} was {acceptance_by_temp[T]:0.4f}.')

# Save vartbl with updated entries
vartbl['samples_by_temp'] = samples_by_temp
vartbl['acceptance_by_temp'] = acceptance_by_temp
save_vartbl(vartbl, fname)


def plots_one_temp(T):
    """Genereate all three plots of interest at one temperature"""
    # Generate the traceplot
    fig1 = plot_trace(trace, f'Trace Plot of x and y at temp. {T}')
    # Generate the autocorr plot
    fig2 = plot_autocorr(trace, f'Autocorrelation Plot of x and y at temp. {T}')
    # Plot the path taken by the sampler
    fig3 = plot_path(samples, x_grid, y_grid, p_grid_by_temp[T], 
                    f'Path Plot for Metropolis Samples @ temp. {T}')
    return [fig1, fig2, fig3]

for T in temps:
    figs = plots_one_temp(T)

# *************************************************************************************************
# B3. Approximate the 𝑓(𝑋) by the appropriate mixture of Gaussians 
# as a way of generating samples from 𝑓(𝑋) to compare with other sampling methods. 
# Use scipy.stats.multivariate_normal to generate 20000 samples. 
# How do the histograms compare with the histograms for the samples from  𝑓(𝑋)  at each temperature. 
# At what temperature do the samples best represent the function?

# *************************************************************************************************
#    # PDF for this multivariate normal as a standalone distribution; see https://docs.pymc.io/prob_dists.html
#    pdf_1 = pm.MvNormal.dist(mu=mu_1.reshape(2,1), cov=cov_1)
#    pdf_2 = pm.MvNormal.dist(mu=mu_2, cov=cov_2)
#    pdf_3 = pm.MvNormal.dist(mu=mu_3, cov=cov_3)
#    # pdf = pm.Mixture.dist(weight, [pdf_1, pdf_2, pdf_3])
#    
#    class MvNormalMixture(pm.Continuous):
#        """Wrapper for a MV normal mixture"""
#        # See https://docs.pymc.io/prob_dists.html, section "Custome distributions"
#        
#        def __init__(self, ws, mus, covs):
#            # check number of data points
#            T = len(ws)
#            assert len(mus) == T
#            assert len(covs) == T
#            self.ws = ws
#            self.mus = mus
#            self.covs = covs
#            self.n = n
#        
#        def random(self, point=None, size=None):
#            return random_samples
#        
#        def logp(self, value):
#            return total_log_prob
#        
#            self.n = len(ws)

