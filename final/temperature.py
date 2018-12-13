"""
Michael S. Emanuel
Thu Dec 13 16:05:11 2018
"""

# core
import numpy as np
from numpy import log
import scipy.stats
# probability
import pymc3 as pm
# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
# import arviz as az
from arviz import plot_trace, plot_autocorr
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
make_cov = lambda  theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

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

cov3 = make_cov(theta3)
sigma3 = np.array([[.4, 0],[0, 1.3]])
mvn3 = scipy.stats.multivariate_normal([3,-2], cov=cov3 @ sigma3 @ cov3.T)

f = lambda xvec: mvn1.pdf(xvec) + mvn2.pdf(xvec) + .5*mvn3.pdf(xvec)

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

# Grid of x and y
grid_size: int = 200
xx = np.linspace(x_min, x_max, grid_size)
yy = np.linspace(y_min, y_max, grid_size)
x_grid, y_grid = np.meshgrid(xx, yy)

# Grid of p(x, y)
try:
    p_grid = vartbl['p_grid']
    print(f'Loaded grid of p(x, y))')
except:    
    p_grid = np.zeros((grid_size, grid_size))
    for i, x in enumerate(xx):
        for j, y in enumerate(yy):
            # Note: it really is p_grid[j, i] and not [i, j]
            p_grid[j, i] = p(x, y)
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
    cs = ax.contour(x_grid, y_grid, p_grid, linewidths=3)
    fig.colorbar(cs, ax=ax)
    ax.grid()
    return fig, ax

# Plot contours of p(x, y)
plot_contour(x_grid, y_grid, p_grid, 'Contours of $p(x, y)$')

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

#    # Assemble into one mixture distribution
weights = np.array([weight_1, weight_2, weight_3])
mus = np.array([mu_1, mu_2, mu_3])

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
# *************************************************************************************************

# Number of samples
num_samples: int = 20000
# Number of dimensions
K: int = 2

# Sample using Metropolis

# The log-probability function
def logp(X: np.array):
    """Log probability at this point"""
    # unpack array X into scalars (x, y)
    (x, y) = X
    return log(p(x, y))

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
            
    return samples, accepted


# Crate a proposal distribution with sigma=1 as per problem statement
def proposal(X, step_size):
    """Normal proposal distribution; step_size is passed from metropolis"""
    return np.random.normal(loc=X, scale=step_size * np.ones(K))


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
    print(f'Loaded metropolis samples')
except:
    # Draw samples
    print(f'Drawing {num_samples_raw} samples with Metropolis algorithm...')
    raw_samples, accepted = metropolis(logp, proposal, step_size, num_samples_raw, X_init)
    # Thin the samples
    samples = raw_samples[tuning::thinning]
    # Save samples to vartbl
    vartbl['raw_samples'] = raw_samples
    vartbl['samples'] = samples
    save_vartbl(vartbl, fname)

# Make trace by hand
trace = dict()
trace['x'] = samples[:,0]
trace['y'] = samples[:,1]

# Plot trace with Arviz
plot_kwargs = {'color':'b'}
plot_trace(trace, trace_kwargs=plot_kwargs)
# Plot autocorr with Arviz
axs = plot_autocorr(trace, max_lag=20)
# ax0, ax1 = axs[0,0], axs[0,1]

def plot_path(samples, x_grid, y_grid, p_grid, title):
    """Generate a path plot"""

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
    return fig, ax

fig, ax = plot_path(samples, x_grid, y_grid, p_grid, 'Path Plot for Metropolis Samples')
