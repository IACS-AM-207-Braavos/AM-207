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
import os
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
temps = df.temp.values
fails = df.fail.values
temps_pass = df[df.fail==0]['temp'].values
temps_fail = df[df.fail==1]['temp'].values

# Plot summary of data
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Shuttle Launches vs. O-Ring Failure')
ax.set_xlabel('Temperature (F)')
ax.set_ylabel('Failure? (1 = Failed)')
ax.scatter(temps_pass, np.zeros_like(temps_pass), label='Pass', color='b')
ax.scatter(temps_fail, np.ones_like(temps_fail), label='Fail', color='r')
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
    def sample_instance(size: int = 1):
        # Preallocate storage; theta is an array of shape (size, 2)
        theta = np.zeros((size, 2))
        # Draw samples for theta_1 and theta_2
        theta[:, 0] = np.random.normal(loc=theta_1_mean, scale=theta_1_std, size=size)
        theta[:, 1] = np.random.normal(loc=theta_2_mean, scale=theta_2_std, size=size)
        # Return the combined theta array
        return theta

    # Return the instance with bound parameter values
    return sample_instance

# Selected parameters for priors of theta_1 and theta_2 are derived from the output of 
# the logistic regression estimate in statsmodels; 
# keep the means unchanged but inflate the standard deviations by 2.0x
theta_1_mean = 15.04
theta_1_std = 7.38*2
theta_2_mean = -0.23
theta_2_std = 0.108*2

# Create instances of the prior and its sampling function with bound parameter values
theta_prior = theta_prior_generic(theta_1_mean, theta_1_std, theta_2_mean, theta_2_std)
sample_theta_prior = sample_theta_prior_generic(theta_1_mean, theta_1_std, theta_2_mean, theta_2_std)


# *************************************************************************************************
# 2.2. Produce 5000-10000 samples from the posterior distribution of Θ using rejection sampling, 
# and plot them and their marginals. (This may take a while.)


# Functions for likelihood, posterior, and grid box for thetas
def sigmoid(z):
    """Sigmoid (logistic) function"""
    return 1.0 / (1.0 + exp(-z))

def likelihood(temps, fails, theta_1, theta_2):
    """Compute the likelihood of the data given parameter values theta_1 and theta_2"""
    # Compute the z score for each launch
    z = theta_1 + theta_2 * temps
    # Compute the sigmoid probabilities of failure at each launch
    fail_prob = sigmoid(z)
    pass_prob = 1.0 - fail_prob
    # The likelihood is the product over every launch of the probability of the predicted events
    pred_prob = fails * fail_prob + (1 - fails) * pass_prob
    return np.prod(pred_prob)


def theta_post(theta_1, theta_2):
    """Posterior probability for theta; uses theta_prior and likelihood defined above"""
    return theta_prior(theta_1, theta_2) * likelihood(temps, fails, theta_1, theta_2)

# Set minimum and maximum for theta_1 and theta_2 using a two standard deviation range
# (3 SD would be better, but it's already going to take a long time...)
theta_1_min, theta_1_max = theta_1_mean - 2*theta_1_std, theta_1_mean + 2*theta_1_std
theta_2_min, theta_2_max = theta_2_mean - 2*theta_2_std, theta_2_mean + 2*theta_2_std
# Package theta limits for readability
theta_limits = (theta_1_min, theta_1_max, theta_2_min, theta_2_max)


# *************************************************************************************************
# 2.2 Estimate L_star (largest value of posterior) by sampling the prior or grid search
def L_star_sample(sample_theta_prior, sample_size):
    """Estimate L_star by sampling"""
    # pre-allocate space for the samples
    post_probs = np.zeros(sample_size)
    # Sample theta from the prior
    thetas = sample_theta_prior(sample_size)
    # Maximum posterior seen so far and accompanying theta    
    max_post_seen = 0.0
    argmax_theta = None
    # Evaluate the posterior probability on sample_size samples
    for i, theta in enumerate(thetas):        
        post_prob = theta_post(theta[0], theta[1])
        post_probs[i] = post_prob
        # Keep running track of the maximum posterior and the associated theta
        if post_prob > max_post_seen:
            max_post_seen = post_prob
            argmax_theta = theta
    # Take three summary statistics over this sample: max, mean, std
    L_max = np.max(post_probs)
    L_mean = np.mean(post_probs)
    L_std = np.std(post_probs)
    # Return summary statistics of posterior probability on this sample
    return L_max, L_mean, L_std, argmax_theta


def L_star_grid(theta_limits, grid_size):
    """Estimate L_star by building a grid"""
    # Unpack theta_limits
    theta_1_min, theta_1_max, theta_2_min, theta_2_max = theta_limits
    # Create a grid of both parameters
    theta_1_samples = np.linspace(theta_1_min, theta_1_max, num=grid_size)
    theta_2_samples = np.linspace(theta_2_min, theta_2_max, num=grid_size)
    # Maximum posterior seen so far and accompanying theta    
    max_post_seen = 0.0
    argmax_theta = None    
    # Evaluate posterior probability on the grid
    post_grid = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        theta_1 = theta_1_samples[i]
        for j in range(grid_size):
            theta_2 = theta_2_samples[j]
            # Save posterior probability on post_grid
            post_prob = theta_post(theta_1, theta_2)
            post_grid[i,j] = post_prob
            # Keep running track of the maximum posterior and the associated theta
            if post_prob > max_post_seen:
                max_post_seen = post_prob
                argmax_theta = np.array([theta_1, theta_2])
    # Take three summary statistics over this sample: max, mean, std
    L_max = np.max(post_grid)
    L_mean = np.mean(post_grid)
    L_std = np.std(post_grid)
    return L_max, L_mean, L_std, argmax_theta

 
# *************************************************************************************************
# Estimate the maximum posterior, L_star, and report it
if 'L_max' not in locals():
    sample_size: int = 10**5
    L_max, L_mean, L_std, argmax_theta = L_star_sample(sample_theta_prior, sample_size)
    work_per_sample = L_max / L_mean
    L_star = L_max
    msg = f'Using {sample_size} samples form the prior, maximum posterior L_max = {L_max:0.3e}, '
    msg += f'L_mean = {L_mean:0.3e}, iterations per sample = {work_per_sample:0.2f}.'
    print(msg)

if 'L_max_grid' not in locals():
    grid_size = 200
    L_max_grid, L_mean_grid, L_std_grid, argmax_theta_grid = L_star_grid(theta_limits, grid_size)  
    work_per_sample_grid = L_max_grid / L_mean_grid
    msg = f'Using a grid of size {grid_size} on the prior, maximum posterior L_max = {L_max_grid:0.3e}, '
    msg += f'L_mean = {L_mean_grid:0.3e}, iterations per sample = {work_per_sample_grid:0.3f}.'
    print(msg)


# *************************************************************************************************
# 2.2 perform the sampling of theta
def sample_theta_post(theta_limits, theta_post, L_star, num_reps):
    """Sample theta_1 and theta_2 on the posterior by rejection sampling.  
    Persist samples in a CSV due to long run time."""
    # Unpack theta_limits
    theta_1_min, theta_1_max, theta_2_min, theta_2_max = theta_limits
    # Create array to store parameters
    thetas = np.zeros((num_reps, 2))    
    # Count both successes and attempts
    idx: int = 0
    attempts: int = 0
    # Status update
    bucket_size: int = 10**4
    buckets: int = 0
    # Start timer
    t0 = time.time()
    # Continue drawing theta until we have num_reps
    while idx < num_reps:
        # Draw two candidates uniformly
        theta_1 = np.random.uniform(theta_1_min, theta_1_max)
        theta_2 = np.random.uniform(theta_2_min, theta_2_max)
        # Compute the posterior
        post = theta_post(theta_1, theta_2)
        # Update L_star
        L_star = max(L_star, post)
        # Rejection sample
        if np.random.uniform(low=0, high=L_star) <= post:
            # Save this sample
            thetas[idx] = (theta_1, theta_2)
            idx += 1
        # Count the attempt
        attempts += 1
        # Status update
        if attempts >= (buckets+1) * bucket_size:
            buckets = attempts // bucket_size
            t1 = time.time()
            # Compute time used
            time_used = (t1 - t0)
            # Estimate remaining time
            time_proj = num_reps / idx * time_used if idx > 0 else np.NAN
            print(f'Completed {buckets} buckets and {idx} samples.', end=' ')
            print(f'Elapsed time {time_used:0.0f}, projected {time_proj:0.0f} seconds.')
    # Return the parameters and the number of attempts
    return (thetas, attempts)

# Test whether the posterior sample of theta was saved to file on a previous run
fname_theta_sample = 'challenger_theta_sample.csv'
if fname_theta_sample in os.listdir():
    # If it was saved, just load it
    df_theta_sample = pd.read_csv(fname_theta_sample, index_col=0)
    theta_sample = df_theta_sample.values
    num_reps = len(df_theta_sample)
    print(f'Loaded {num_reps} samples of theta drawn from posterior distribution.')
else:
    # It it wasn't save, draw samples from the posterior distribution and save them
    num_reps = 10000
    theta_sample, attempts = sample_theta_post(theta_limits, theta_post, L_star, num_reps)
    print(f'Generated {num_reps} samples from posterior distribution of theta in {attempts} attempts.')
    # Save the sampled thetas into a DataFrame
    df_theta = pd.DataFrame(data=theta_sample, columns=['theta_1', 'theta_2'])
    # Save it if not already present
    df_theta.to_csv(fname_theta_sample)

# Alias samples for theta_1 and theta_2 for legibility
theta_1_sample = theta_sample[:, 0]
theta_2_sample = theta_sample[:, 1]

# *************************************************************************************************
# 2.2 Plot the sampled thetas and their marginals

# Plot the samples thetas (joint distribution)
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title(f'{num_reps} Samples from Posterior Distribution')
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')
ax.grid()
ax.scatter(theta_1_sample, theta_2_sample, s=6, alpha=0.5)
plt.show()

# Plot the marginal distribution of theta_1
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title(r'Marginal Distribution of $\theta_1$')
ax.set_xlabel(r'$\theta_1$')
ax.hist(theta_1_sample, bins=60)
# Overlay the 68% confidence interval (hard code these here)
ax.axvline(x=7.705, color='r')
ax.axvline(x=22.381, color='r')
ax.grid()
plt.show()


# Plot the marginal distribution of theta_2
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title(r'Marginal Distribution of $\theta_2$')
ax.set_xlabel(r'$\theta_2$')
ax.hist(theta_2_sample, bins=60)
# Overlay the 68% confidence interval (hard code these here)
ax.axvline(x=-0.340, color='r')
ax.axvline(x=-0.125, color='r')
ax.grid()
plt.show()

# *************************************************************************************************
# 2.3. Use the logit package in the statsmodels library to compute 68% confidence intervals on the θ parameters. 
# Compare those intervals with the 68% credible intervals from the posterior above. 
# Overlay these on the above marginals plots.

# Create an array X of predictors for statsmodel including a constant
X = sm.add_constant(df.temp)
# Fit logit model
logit_model = Logit(endog=df.fail, exog=X).fit()
# Report the model summary
print(logit_model.summary2())

# Get the parameter values from the logit model
logit_thetas = logit_model.params
# Extract the 68% confidence intervals
conf_intervals = logit_model.conf_int(alpha=0.32)
print(f'Parameter estiamtes from Logistic Regression Model (68% confidence intervals)')
print(f'68% confidence interval for theta_1 = {conf_intervals.values[0, 0]:0.3f} - {conf_intervals.values[0, 1]:0.3f}')
print(f'68% confidence interval for theta_2 = {conf_intervals.values[1, 0]:0.3f} - {conf_intervals.values[1, 1]:0.3f}')
# Get the 16th and 84th percentiles for theta_1 and theta_2
theta_1_cred = np.percentile(theta_sample[:, 0], [16, 84])
theta_2_cred = np.percentile(theta_sample[:, 1], [16, 84])
print(f'\nParameter estimates from Posterior Sampling (68% credible intervals)')
print(f'68% credible interval for theta_1   = {theta_1_cred[0]:0.3f} - {theta_1_cred[1]:0.3f}')
print(f'68% credible interval for theta_2   = {theta_2_cred[1]:0.3f} - {theta_2_cred[1]:0.3f}')


# *************************************************************************************************
# 2.4. Use the MLE values from statsmodels and the posterior mean from 2.2 at each temperature to plot the probability
#  of failure in the frequentist and bayesian settings as a function of temperature. What do you see?

def fail_prob(theta_1, theta_2, temp):
    """The probability of failure at this temperature given theta_1 and theta_2"""
    z = theta_1 + theta_2 * temp
    return sigmoid(z)

# Vector of temperatures to test
plot_temp = arange_inc(20, 100)
# Alias the MLE estimates for legibility
theta_1_mle, theta_2_mle = logit_thetas
# Compute the posterior means for theta_1 and theta_2
theta_1_pm = np.mean(theta_1_sample)
theta_2_pm = np.mean(theta_2_sample)
# Compute the failure probability vs. temperature in both models
plot_fail_mle = fail_prob(theta_1_mle, theta_2_mle, plot_temp)
plot_fail_pm = fail_prob(theta_1_pm, theta_2_pm, plot_temp)

# Plot the two curves of failure probability vs. temperature
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Failure Probability vs. Temperature - 2 Models')
ax.set_xlabel('Temperature, Degrees Fahrenheit')
ax.set_ylabel('Failure Probability')
ax.plot(plot_temp, plot_fail_mle, label='MLE Params')
ax.plot(plot_temp, plot_fail_pm, label='Post. Mean')
ax.legend()
ax.grid()
plt.show()


# *************************************************************************************************
# 2.5. Compute the mean posterior probability for an O-ring failure at t=31∘F. 
# To do this you must calculate the posterior at 31∘F and take the mean of the samples obtained.

# Set the temperature
temp_cold = 31
# Compute the failure probability using the posterior mean parameter estimates
fail_prob_pm = fail_prob(theta_1_pm, theta_2_pm, temp_cold)
print(f'The posterior mean probability of failure for a cold launch at {temp_cold} Fahrenheit '
      f'is {fail_prob_pm*100:0.1f}%')
 

# *************************************************************************************************
# 2.6. You can instead obtain the probability from the posterior predictive. 
# Use the posterior samples to obtain samples from the posterior predictive at 31∘F 
# and calculate the fraction of failures.

# For each pair (theta_1, theta_2) compute the failure probability at the cold temperature
# Then take the mean of these posterior predictives.
fail_prob_pp = np.mean([fail_prob(theta_1, theta_2, temp_cold) for theta_1, theta_2 in theta_sample])
print(f'The posterior predictive probability of failure for a cold launch at {temp_cold} Fahrenheit '
      f'is {fail_prob_pp*100:0.1f}%')

# *************************************************************************************************
# 2.7. The day before a new launch, meteorologists predict that the temperature will be T∼N(68,1) during take-off. 
# Estimate the probability for an O-ring failure during this take-off. 
# (You will calculate multiple predictives at different temperatures for this purpose).

# Take 10000 samples of the temperature
temp_forecast_samples = np.random.normal(loc=68, scale=1,size=num_reps)
# Compute 10000 failure probabilities, each one a draw of theta from the posterior against a temperature draw
fail_probs_fc = np.zeros(num_reps)
for i in range(num_reps):
    # Get paired theta values and temperature in this simulation
    theta_1, theta_2 = theta_sample[i]
    temp = temp_forecast_samples[i]
    fail = fail_prob(theta_1, theta_2, temp)
    fail_probs_fc[i] = fail
# The estimated probability is the mean of these sampled probabilities
fail_prob_fc = np.mean(fail_probs_fc)
# Report results
print(f'Estimated mean failure probability for a launch with temperature forecast to be '\
      f'normal with mean=68, std dev = 1 is {fail_prob_fc*100:0.1f}%.')