"""
Harvard IACS AM 207
Homework 5
Problem 3

Michael S. Emanuel
Wed Oct 10 23:53:53 2018
"""

import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt
from am207_utils import plot_style

# Set plot style
plot_style()

# *************************************************************************************************
# Question 3: Maximum Uniformity -- Frequentist Bootstraps and the Bayesian Posterior
# *************************************************************************************************

# Recall in HW 3 Question 1 we attempted to explore an edge case in using non-parametric bootstrap to construct 
# confidence intervals. Let's revisit the setup of that problem. 
# Suppose you have {X1,X2,...Xn} datapoints such that Xi are independently and identically drawn from a 
# Unif(0,θ). Consider the extreme order statistic Y = X(n) = max(X1,X2,...Xn).


# *************************************************************************************************
# 3.1. Derive (or possibly re-write from HW3) expressions for F_Y(y|n,θ) the CDF of Y and f_Y(y|n,θ) the pdf of Y.
# See notebook

# *************************************************************************************************
# 3.2. In HW3 we had difficulty constructing confidence intervals to estimate θ using our normal percentiles 
# methods so instead we introduced pivot confidence intervals. 
# Let's reframe the problem so that we can use percentiles to construct our confidence intervals. 
# Define Z≡n⋅(θ−Y) and use elementary calculation to write an expression for 
# F_Z(z|n,θ) the CDF of Z and f_Z(z | n,θ) the pdf of Z.
# See notebook

# *************************************************************************************************
# 3.3. What is the limiting distribution of Z (as n→∞)? Plot that limiting distribution.
# See notebook

# 3.3 Plot the exponential distribution

# Generate an example exponential distribution
plot_x = np.arange(0, 6, 0.05)
plot_pdf = expon.pdf(plot_x, loc=0, scale=1)
plot_cdf = expon.cdf(plot_x, loc=0, scale=1)

# Plot the PDF
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Limiting Distribution of $f_Z(Z)$, the Exponential')
ax.set_xlabel('Z')
ax.set_ylabel('PDF')
ax.plot(plot_x, plot_pdf)
ax.grid()
plt.show()

# Plot the CDF
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Limiting Distribution of $F_Z(Z)$, the Exponential')
ax.set_xlabel('Z')
ax.set_ylabel('CDF')
ax.plot(plot_x, plot_cdf)
ax.grid()
plt.show()


# *************************************************************************************************
# 3.4. Use scipy/numpy to generate 100000 samples {Xi} from Unif(0,100) (i.e. let θ = 100). 
# Store them in Based on your data sample, what's θ̂ the empirical estimate for θ.

# Number of samples requested
sample_size = 100000
# Given value of theta
theta = 100
# Set the random seed
np.random.seed(42)
# Build samples with numpy
x_samples = np.random.uniform(low=0.0, high=theta, size=sample_size)
# The empirical estimate is just the maximum
theta_hat_ml = np.max(x_samples)
print(f'Sample size = {sample_size}, ML estimate of theta = {theta_hat_ml:0.6f}.')

# *************************************************************************************************
# 3.5. Use non-parametric bootstrap to generate a sampling distribution of 10000 estimates for Z by substituting 
# θ^ for θ. Plot a histogram of your sampling distribution. Make sure to title and label the plot. 
# Use percentiles to construct the 10% and 68% bootstrap confidence intervals. Plot them in your graph.
# Hint: Should the confidence intervals be symmetric around the estimate θ^?

# Generate samples of z with the ML estimate of theta
z_samples = np.random.exponential(scale=theta_hat_ml, size=10000)
# Plot the histogram
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Non-Parametric Bootstrap - Sampling Distribution of $Z$')
ax.set_xlabel('Z')
ax.set_ylabel('Probability')
ax.hist(z_samples, bins=100, density=True)
ax.grid()
plt.show()


# *************************************************************************************************
# 3.6. Make an argument that we can construct a bootstrap confidence interval that 
# always mismatches the limiting distribution.

# See notebook

# *************************************************************************************************
# 3.7. Let's switch to being Bayesian. In 3.1 we came up with an expression for the likelihood 
# fY(y | n,θ)fY(y | n,θ). Use the Pareto distribution to construct a prior for θ. 
# What are some reasonable values to use for the scale and shape?

# See notebook
# theta_m = y (minimum theta is maximum data point seen)
# alpha = 1 (one pseudo-observation)
# f(theta) = y / theta^2 on [y, infinity)

# *************************************************************************************************
# 3.8. Write down an expression for the posterior distribution f_Y(θ | n,y)

# See notebook
# f_post(theta) = ny / theta^(n+1) on [y, infinity)

# *************************************************************************************************
# 3.9. Draw 10000 posterior samples and plot a histogram of the posterior distribution. 
# Use percentiles to construct the 68% HPD. Plot the posterior distribution and mark the HPD on your plot.

# Number of samples
n: int = len(x_samples)
# Draw 10,000 samples from the posterior, a pareto with shape=n+1 and scale = theta_hat
# As per the numpy documentation, the library function is for the modified Pareto distribution
# We need to add 1 and multiply it by x_m to recover the Pareto distribution we want
theta_samples = theta_hat_ml * (1+np.random.pareto(a=n+1, size=10000))
# The 68% range spans percentiles 16 to 84
theta_lo, theta_hi = np.percentile(theta_samples, q=[16, 84])
# Plot the posterior distribution with the HPD marked
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title(r'Poseterior Samples of $\theta$')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel('Probability')
# Turn off axis shifting so we can see a regular scale
ax.get_xaxis().get_major_formatter().set_useOffset(False)
ax.hist(theta_samples, bins=100, density=True)
ax.axvline(theta_lo, color='r')
ax.axvline(theta_hi, color='r')
ax.grid()
plt.show()

# *************************************************************************************************
# 3.10. How does the 68% HPD compare with the confidence interval generated from bootstrapping? 
# Why doesn't the bayesian interval construction suffer the same concerns you noted in 3.6