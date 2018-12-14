"""
Michael S. Emanuel
Thu Dec 13 16:05:11 2018
"""

# core
import numpy as np
from numpy import log, exp
import scipy.stats
from scipy.stats import multivariate_normal
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
from typing import List, Dict, Callable

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
mu_1 = np.array([12, 7])
R_1 = make_cov(theta1)
cov_1 = R_1 @ sigma1 @ R_1.T
weight_1 = 1.0 / 2.5
#    
# mu, sigma, and weight for Gaussian 2
mu_2 = np.array([-1, 6])
R_2 = make_cov(theta2)
cov_2 = R_2 @ sigma2 @ R_2.T
weight_2 = 1.0 / 2.5

# mu, sigma, and weight for Gaussian 3
mu_3 = np.array([3, -2])
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
    # Draw num_samples sample points
    iterator_i = tqdm.tqdm(range(num_samples)) if num_samples > 1000 else range(num_samples)
    for i in iterator_i:
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
X_init = np.average(mus, axis=0, weights=weights)
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
# fig = plot_trace(trace, 'Trace Plot of x and y')
# Generate the autocorr plot
# fig = plot_autocorr(trace, 'Autocorrelation Plot of x and y')
# Plot the path taken by the sampler
# fig = plot_path(samples, x_grid, y_grid, p_grid, 'Path Plot for Metropolis Samples')

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
    # plot_contour(x_grid, y_grid, p_grid_T, f'Contours of $p(x, y)$ at temperature T={T}')
    pass


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
    """Generate all three plots of interest at one temperature"""
    # Get the trace for this temperature
    trace_T = {'x': samples_by_temp[T][:,0],
               'y': samples_by_temp[T][:,1]}
    # Generate the traceplot
    fig1 = plot_trace(trace_T, f'Trace Plot of x and y at temp. {T}')
    # Generate the autocorr plot
    fig2 = plot_autocorr(trace_T, f'Autocorrelation Plot of x and y at temp. {T}')
    # Plot the path taken by the sampler
    fig3 = plot_path(samples_by_temp[T], x_grid, y_grid, p_grid_by_temp[T], 
                    f'Path Plot for Metropolis Samples @ temp. {T}')
    return [fig1, fig2, fig3]

for T in temps:
    # figs = plots_one_temp(T)
    pass

# *************************************************************************************************
# B3. Approximate the 𝑓(𝑋) by the appropriate mixture of Gaussians 
# as a way of generating samples from 𝑓(𝑋) to compare with other sampling methods. 
# Use scipy.stats.multivariate_normal to generate 20000 samples. 
# How do the histograms compare with the histograms for the samples from  𝑓(𝑋)  at each temperature. 
# At what temperature do the samples best represent the function?

# Draw a matrix of samples from all three Guassians
samples_1 = multivariate_normal(mean=mu_1, cov=cov_1).rvs(num_samples)
samples_2 = multivariate_normal(mean=mu_2, cov=cov_2).rvs(num_samples)
samples_3 = multivariate_normal(mean=mu_3, cov=cov_3).rvs(num_samples)
# Stack the candidate samples; 
samples_cand = np.stack([samples_1, samples_2, samples_3])

# Sample ancestrally (first pick a Guassian, then use that column)
cluster = np.random.choice(3, size=num_samples, p=weights)
samples_gm = samples_cand[cluster,range(num_samples),:]

def plots_all(samples, model_name):
    """Generate all three plots of interest from the Gaussian Mixture"""
    # Get the trace for this temperature
    trace = {'x': samples[:,0],
             'y': samples[:,1]}
    # Generate the traceplot
    fig1 = plot_trace(trace, f'Trace Plot of x and y from {model_name}')
    # Generate the autocorr plot
    fig2 = plot_autocorr(trace, f'Autocorrelation Plot of x and y from {model_name}')
    # Plot the path taken by the sampler
    fig3 = plot_path(samples, x_grid, y_grid, p_grid, 
                    f'Path Plot for {model_name}')
    return [fig1, fig2, fig3]

# *************************************************************************************************
# Part C: Parallel Tempering
# Now that we've seen some of the properties of sampling at higher temperatures, let's explore 
# a way to incorporate the improved exploration of the entire pdf from sampling at higher temperatures 
# while still getting samples that match our distribution. 
# We'll use a technique called parallel tempering.
# *************************************************************************************************

# The general idea of parallel tempering is to simulate 𝑁 replicas of the original system of interest 
# (in our case, a single Metropolis Hastings chain), each replica at a different temperature. 
# The temperature of a Metropolis Hastings Markov Chain defines how likely 
# it is to sample from a low-density  part of the target distribution. 
# The high temperature systems are generally able to sample large volumes of parameter space, 
# whereas low temperature systems, while having precise sampling in a local region of parameter space, 
# may become trapped around local energy minima/probability maxima. 
# Parallel tempering achieves good sampling by allowing the chains at different temperatures to 
# exchange complete configurations. 
# Thus, the inclusion of higher temperature chains ensures that the lower temperature chains can access 
# all the low-temperature regions of phase space: the higher temperatures help these chains make the jump-over.

# Here is the idea that you must implement.
# There are 𝑁 replicas each at different temperatures 𝑇𝑖 that produce 𝑛 samples each before possibly swapping states.
# We simplify matters by only swapping states at adjacent temperatures. 
# The probability of swapping any two instances of the replicas is given by

# 𝐴=min(1.0, p_k(
# One of the 𝑇𝑖's in our set will always be 1 and this is 
# the only replica that we use as output of the Parallel tempering algorithm.

# An algorithm for Parallel Tempering is as follows:
# Initialize the parameters {(𝑥𝑖𝑛𝑖𝑡,𝑦𝑖𝑛𝑖𝑡)𝑖},{𝑇𝑖},𝐿 where
# 𝐿 is the number of iterations between temperature swap proposals.
# {𝑇𝑖} is a list of temperatures. You'll run one chain at each temperature.
# {(𝑥𝑖𝑛𝑖𝑡,𝑦𝑖𝑛𝑖𝑡)𝑖} is a list of starting points, one for each chain
# For each chain (one per temperature) use the simple Metropolis code you wrote earlier. 
# Perform 𝐿 transitions on each chain.
# Set the {(𝑥𝑖𝑛𝑖𝑡,𝑦𝑖𝑛𝑖𝑡)𝑖} for the next Metropolis run on each chain to the last sample for each chain i.
# Randomly choose 2 chains at adjacent temperatures.
# Use the above formula to calculate the Acceptance probability 𝐴.
# With probability 𝐴, swap the positions between the 2 chains (that is swap the 𝑥s of the two chains, 
# and separately swap the 𝑦s of the chains .
# Go back to 2 above, and start the next L-step epoch
# Continue until you finish 𝑁𝑢𝑚.𝑆𝑎𝑚𝑝𝑙𝑒𝑠//𝐿 epochs.


# *************************************************************************************************
# C1. Explain why swapping states with the given acceptance probability is in keeping with detailed balance. 
# in notebook


# *************************************************************************************************
# C2. Create a parallel tempering sampler that uses 5 chains at the temperatures {0.1, 1, 3, 7, 10} 
# to sample from 𝑓(𝑥,𝑦). 
# Choose a value of L around 10-20. 
# Generate 10000 samples from 𝑓(𝑥,𝑦). 
# Construct histograms of the marginals, traceplots, autocorrelation plots, and a pathplot for your samples.

def metropolis_temp(pdf: Callable, T: float, num_samples: int, X_init: np.ndarray):
    """Modified metropolis sampler that accept a pdf and temperature"""
    # create p(X;T) in situ by binding T
    logp_T = lambda X : log(pdf(X)) / T
    # Draw samples
    samples, accepted = metropolis(logp_T, proposal, step_size, num_samples, X_init)
    # Only return the samples; discard number of accepted points
    return samples


def trans_prob(pdf, Xi: np.ndarray, Xj: np.ndarray, Ti: float, Tj: float):
    """
    Compute the transition probability A(i, j) between two states
    pdf: probability density function taking vectorized input
    Xi: the first point, corresponding to sampler with temperature Ti
    Xj: the second point, corresponding to sampler with temperature Tj
    Ti:  temperature of the first point, Ti
    Tj:  temperature of the second pont, Tj
    """
    # Evaluate the basic probability (without temperature) at both points
    pi = pdf(Xi)
    pj = pdf(Xj)
    # Compute the four probabilities appearing in the formula    
    pi_xi = np.power(pi, 1.0 / Ti)
    pi_xj = np.power(pi, 1.0 / Tj)
    pj_xi = np.power(pj, 1.0 / Ti)
    pj_xj = np.power(pj, 1.0 / Tj)
    # Apply the formula for A
    return min(1.0, (pi_xj * pj_xi) / (pi_xi * pj_xj))


def swap_inits(X_inits: List[np.ndarray], i: int, j: int):
    """Swap the states (initializers) for two chains"""
    # Copy the starting states
    init_i, init_j = X_inits[i].copy(), X_inits[j].copy()
    # Swap them
    X_inits[i] = init_j
    X_inits[j] = init_i


def parallel_temper(pdf: Callable, temps: np.ndarray, X_init, L: int, epochs: int):
    """Parallel tempering algorithm"""
    # Get the number of parallel chains
    num_chains: int = len(temps)
    # Shape of data
    K: int = X_init.shape[0]
    # Compute the total number of samples and preallocate space
    # Will only save the results from the chain with temperature=1
    num_samples: int = L * epochs
    samples = np.zeros((num_samples, K))
    # Get index of the chain with temperature T=1
    idx_t0: int = np.searchsorted(temps, 1.0)
    assert temps[idx_t0] == 1.0, 'Error: one of the temps must be 1.0!'
    
    # Bind arguments with the PDF and L to metropolis_temp
    make_chain = lambda X_start, T : metropolis_temp(pdf, T, L, X_start)
    # Save range of i's for legibility
    ii = range(num_chains)    
    # Initialize X_inits
    X_inits = [X_init.copy() for i in ii]
    # initialize over epochs
    for epoch in tqdm.tqdm(range(epochs)):
        # Run new chains
        chains = [make_chain(X_inits[i], temps[i]) for i in ii]
        # Copy output from chain at temp 0 to samples
        i0: int = L*(epoch+0)
        i1: int = L*(epoch+1)
        samples[i0:i1] = chains[idx_t0]
    
        # Copy current states of each chain to X_starts
        X_inits = [chain[-1] for chain in chains]
        # Randomly choose a pair of adjacent chains; equivalent to picking k in [0, num_chains-1)
        k = np.random.randint(num_chains-1)
        # Extract Xi, Xj, Ti, Tj for transition probability
        Xi = X_inits[k]
        Xj = X_inits[k+1]
        Ti = temps[k]
        Tj = temps[k+1]
        # Compute the transition probability
        tp: float = trans_prob(pdf, Xi, Xj, Ti, Tj)
        # Draw a random uniform to determine whether there is a transition
        if np.random.random() < tp:
            swap_inits(X_inits, k, k+1)

    # Return only the "good" samples from the chain with temp-1
    return samples
        
# Alias the vectorized probability desnity function for legibility
pdf = f
# Set parameters for paralllel tempering
temps_pt = np.array([0.1, 1.0, 3.0, 7.0, 10.0])
num_samples_pt: int = 10000
L: int = 16
epochs = int(np.ceil(num_samples_pt / L))
# Run parallel tempering
try:
    samples_pt = vartbl['samples_pt']
    print(f'Loaded {len(samples_pt)} samples from parallel tempering.')
except:
    print(f'Generating {num_samples_pt} samples with parallel tempering...')
    samples_pt = parallel_temper(pdf, temps_pt, X_init, L, epochs)
    vartbl['samples_pt'] = samples_pt
    save_vartbl(vartbl, fname)



# *************************************************************************************************
# C3. How do your samples in C2 compare to those of the Metropolis sampler? 
# How do they compare to the samples generated from the Gaussian Mixture approximation of f(x)?

