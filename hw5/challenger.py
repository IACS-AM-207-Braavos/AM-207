"""
Michael S. Emanuel
Tue Oct  9 22:14:15 2018
"""

import numpy as np
from numpy import exp
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
import pandas as pd
import matplotlib.pyplot as plt
import time
from am207_utils import plot_style, arange_inc

# Set plot style
plot_style()

# *************************************************************************************************
# Question 2: The Consequences of O-ring Failure can be Painful and Deadly
# *************************************************************************************************

# In 1986, the space shuttle Challenger exploded during take off, killing the seven astronauts aboard. 
# It is believed that the explosion was caused by the failure of an O-ring 
# (a rubber ring that seals parts of the solid fuel rockets together), 
# and that the failure was caused by the cold weather at the time of launch (31F).

# In the file chall.txt, you will find temperature (in Fahrenheit) and failure data from 23 shuttle launches, 
# where 1 stands for O-ring failure and 0 no failure. We assume that the observed temperatures are fixed and that,  
# at temperature tt, an O-ring fails with probability f(θ1+θ2t)conditionally on Θ=(θ1,θ2)

# f(z) is defined to be the logistic function -- f(z)=1/(1+exp(−z))

# *************************************************************************************************
# 2. Visualize the data
# Download the file into a DataFrame
df = pd.read_csv('chall.txt', names=['temp', 'fail'], delim_whitespace=True, index_col=False)
# Alias into local variables
temp = df.temp.values
fail = df.fail.values
temp_pass = df[df.fail==0]['temp'].values
temp_fail = df[df.fail==1]['temp'].values

# Plot summary of data
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Shuttle Launches vs. O-Ring Failure')
ax.set_xlabel('Temperature (F)')
ax.set_ylabel('Failure? (1 = Failed)')
ax.scatter(temp_pass, np.zeros_like(temp_pass), label='Pass', color='b')
ax.scatter(temp_fail, np.ones_like(temp_fail), label='Fail', color='r')
ax.legend(loc='lower left')
ax.grid()


# *************************************************************************************************
# 2.1. Based on your own knowledge and experience, suggest a prior distribution for the regression parameters (Θ1,Θ2). 
# Make sure to explain your choice of prior.

def theta_prior_generic(theta_1_mean, theta_1_std, theta_2_mean, theta_2_std):
    """
    Make a generic prior functrion that wraps the means and standard deviations
    theta_2_mean:       the mean of theta_1 on a normal distribution
    theta_1_std:        the standard deviation of theta_1 on a normal distribution
    theta_2_mean:       the mean of theta_2 on a normal distribution
    theta_2_std:        the standard deviation of theta_2 on a normal distribution
    """
    # Return a function of two variables
    def prior_instance(theta_1, theta_2):
        # Probability for theta_1
        theta_1_prob = norm.pdf(theta_1, loc=theta_1_mean, scale=theta_1_std)
        theta_2_prob = norm.pdf(theta_2, loc=theta_2_mean, scale=theta_2_std)
        # Return the joint probability
        return theta_1_prob * theta_2_prob
    
    # Return this function with the bound parameter values
    return prior_instance


def sample_theta_prior_generic(theta_1_mean, theta_1_std, theta_2_mean, theta_2_std):
    """
    Draw samples for theta_1 and theta_2 on the prior distribution
    theta_2_mean:       the mean of theta_1 on a normal distribution
    theta_1_std:        the standard deviation of theta_1 on a normal distribution
    theta_2_mean:       the mean of theta_2 on a normal distribution
    theta_2_std:        the standard deviation of theta_2 on a normal distribution
    """
    def sample_instance(size: int):
        # Preallocate storage; theta is an array of shape (size, 2)
        theta = np.zeros((size, 2))
        # Draw samples for theta_1 and theta_2
        theta[:, 0] = np.random.normal(loc=theta_1_mean, scale=theta_1_std, size=size)
        theta[:, 1] = np.random.normal(loc=theta_2_mean, scale=theta_2_std, size=size)
        # Return the combined theta array
        return theta

    # Return the instance with bound parameter values
    return sample_instance

# Selected parameters for priors of theta_1 and theta_2
theta_1_mean = 15.0
theta_1_std = 15.0
theta_2_mean = -0.22
theta_2_std = 0.22

# Create instances of the prior and its sampling function with bound parameter values
theta_prior = theta_prior_generic(theta_1_mean, theta_1_std, theta_2_mean, theta_2_std)
sample_theta_prior = sample_theta_prior_generic(theta_1_mean, theta_1_std, theta_2_mean, theta_2_std)


# *************************************************************************************************
# 2.2. Produce 5000-10000 samples from the posterior distribution of Θ using rejection sampling, 
# and plot them and their marginals. (This may take a while.)

def sigmoid(z):
    """Sigmoid (logistic) function"""
    return 1.0 / (1.0 + exp(-z))

def likelihood(temp, fail, theta_1, theta_2):
    """Compute the likelihood of the data given parameter values theta_1 and theta_2"""
    # Compute the z score for each launch
    z = theta_1 + theta_2 * temp
    # Compute the sigmoid probabilities of failure at each launch
    fail_prob = sigmoid(z)
    pass_prob = 1.0 - fail_prob
    # The likelihood is the product over every launch of the probability of the predicted events
    pred_prob = fail * fail_prob + (1 - fail) * pass_prob
    return np.prod(pred_prob)


def theta_post(theta_1, theta_2):
    """Posterior probability for theta; uses theta_prior and likelihood defined above"""
    return theta_prior(theta_1, theta_2) * likelihood(temp, fail, theta_1, theta_2)


# Set minimum and maximum for theta_1 and theta_2 using a two standard deviation range
# (3 SD would be better, but it's already going to take a long time...)
theta_1_min, theta_1_max = theta_1_mean - 2*theta_1_std, theta_1_mean + 2*theta_1_std
theta_2_min, theta_2_max = theta_2_mean - 2*theta_2_std, theta_2_mean + 2*theta_2_std

# Create a grid of both parameters
grid_size: int = 400
theta_1_samples = np.linspace(theta_1_min, theta_2_max, num=grid_size)
theta_2_samples = np.linspace(theta_2_min, theta_2_max, num=grid_size)
# theta_1_grid, theta_2_grid = np.meshgrid(theta_1_samples, theta_2_samples)

# Compute posterior on the grid
if 'post_grid' not in globals():
    post_grid = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            post_grid[i,j] = theta_post(theta_1_samples[i], theta_2_samples[j])

# Estimate the maximum posterior, L_star, and report it
L_star = np.max(post_grid)
print(f'Using grid size {grid_size}, maximum posterior L_star = {L_star:0.3e}.')

# *************************************************************************************************
def sample_theta_posterior_OLD(theta_prior, temp, fail, num_reps: int):
    """Sample theta_1 and theta_2 on the posterior by rejection sampling."""
    # Create array to store parameters
    thetas = np.zeros((num_reps, 2))
    # Count both successes and attempts
    idx: int = 0
    attempts: int = 0
    # Status update
    bucket_size: int = 10**6
    buckets: int = 0
    # Start timer
    t0 = time.time()
    # Continue drawing theta until we have num_reps
    while idx < num_reps:
        # Draw two candidates
        theta_1, theta_2 = theta_prior()
        # Compute the likelihood
        like = likelihood(temp, fail, theta_1, theta_2)
        # Rejection sample
        if np.random.uniform() <= like:
            # Save this sample
            thetas[idx] = (theta_1, theta_2)
            idx += 1
        # Count the attempt
        attempts += 1
        # Status update
        if attempts >= (buckets+1) * bucket_size:
            buckets = attempts // bucket_size
            t1 = time.time()
            # Compute time in MINUTES
            time_used = (t1 - t0) / 60.0
            time_proj = num_reps / idx * time_used if idx > 0 else np.NAN
            print(f'Completed {buckets} buckets and {idx} samples.', end=' ')
            print(f'Elapsed time {time_used:0.0f}, projected {time_proj:0.0f} (minutes).')
    # Return the parameters and the number of attempts
    return (thetas, attempts)

# Draw samples from the posterior distribution
# thetas, attempts = sanple_theta_posterior(theta_prior, temp, fail, num_reps=100)


# 2.3. Use the logit package in the statsmodels library to compute 68% confidence intervals on the θ parameters. 
# Compare those intervals with the 68% credible intervals from the posterior above. 
# Overlay these on the above marginals plots.

# Create an array X of predictors for statsmodel including a constant
X = sm.add_constant(df.temp)
# Fit logit model
logit_model = Logit(endog=df.fail, exog=X).fit()
# Report the model summary
print(logit_model.summary2())
# Get the parameter values 
logit_thetas = logit_model.params
# Extract the confidence intervals
conf_intervals = logit_model.conf_int()

# 2.4. Use the MLE values from statsmodels and the posterior mean from 2.2 at each temperature to plot the probability
#  of failure in the frequentist and bayesian settings as a function of temperature. What do you see?


# 2.5. Compute the mean posterior probability for an O-ring failure at t=31∘Ft=31∘F. 
# To do this you must calculate the posterior at 31∘F31∘F and take the mean of the samples obtained.


# 2.6. You can instead obtain the probability from the posterior predictive. 
# Use the posterior samples to obtain samples from the posterior predictive at 31∘F 
# and calculate the fraction of failures.


# 2.7. The day before a new launch, meteorologists predict that the temperature will be T∼N(68,1) during take-off. 
# Estimate the probability for an O-ring failure during this take-off. 
# (You will calculate multiple predictives at different temperatures for this purpose).