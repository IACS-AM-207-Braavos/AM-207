"""
Harvard AM 207
Homework 8
Problem 2

Michael S. Emanuel
Thu Nov  1 23:11:34 2018
"""

import numpy as np
from numpy import sqrt
import pandas as pd
from matplotlib import pyplot as plt
# from scipy.special import gamma, binom


def load_data():
    # Load the data
    df = pd.read_csv('kcancer.csv')
    # Death count: y[j]
    ys = df['dc'].values
    # Population of each county: n[j]
    ns = df['pop'].values
    # Mortality rate of each county
    mortality_rates = df['pct_mortality'].values
    # Return the data frame and the extracted arrays
    return df, ys, ns, mortality_rates


def summary_stats(ns, mortality_rates):
    """Generate summary statistics for the data set"""
    # Sample mean and variance over counties
    # Use correct weights for mean rate (i.e. pool)
    mu = np.average(mortality_rates, weights=ns)
    # Sample variance over counties - this time equally weight the counties
    sigma2 = np.var(mortality_rates)
    
    # Mean of reciprocal size
    recip_size = np.mean(1.0 / ns)

    # Pooled variance and recoprocal size
    # np.average((mortality_rates-mu)**2, weights=ns)
    # recip_size_pooled = np.average(1.0/ns, weights=ns)
    
    # Return the relevant stats
    return mu, sigma2, recip_size
    

def empirical_bayes(mu, sigma2, recip_size):
    """Fit the hyperparameters alpha and beta using empirical bayes"""
    # Implied parameters alpha and beta
    beta = 5.0 / (sigma2 / mu - recip_size)
    alpha = mu * beta / 5.0
    return (alpha, beta)


def recovered_moments(alpha, beta, ns):
    # Implied parameters p and r in negative binomial distribution
    ps = 5*ns / (5*ns + beta)
    r = alpha
    
    # Implied expected values for each county - these are COUNTS, not rates
    expected_counts = ps / (1.0 - ps) * r
    expected_rates = expected_counts / ns
    # Expected variance - by count and rate
    expected_vars_count = ps / (1.0 - ps)**2 * r
    expected_vars_rate = expected_vars_count / ns**2
    
    # Recovered sample mean
    mu_rec = np.mean(expected_rates)
    sigma2_rec = np.mean(expected_vars_rate)

    return (mu_rec, sigma2_rec)


def report_results(ns, mu, sigma2, alpha, beta, mu_rec, sigma2_rec):
    """Report results with summary statistics and model clibration"""
    # Total number of people
    nTotal = int(np.sum(ns))

    # Standard deviations from the variance (empirical and recovered)
    sigma = sqrt(sigma2)
    sigma_rec = sqrt(sigma2_rec)

    # Report summary statistics
    mortality_rate = mu
    print(f'Total number of people: {nTotal}.')
    print(f'Overall cancer mortality rate:     {mortality_rate:0.6f}')
    print(f'Overall cancer mortality approximately one in {int(round(1/mortality_rate))}')
    print(f'Population mean mu = {mu:0.6f}')
    print(f'Population variance sigma2 = {sigma2:0.3e}')
    print(f'Population std dev sigma   = {sigma:0.6f}')

    # Report model calibration results
    print(f'\nEstimated Model Parameters for Gamma prior by moment matching:')
    print(f'alpha={alpha:0.6f}')
    print(f'beta={beta:0.6f}')

    # Check that these parameters recover mu and sigma2
    print(f'Recovered parameter values:')
    print(f'mu = {mu_rec:0.6f}')
    print(f'sigma2 = {sigma2_rec:0.3e}')
    print(f'sigma = {sigma_rec:0.6f}')


# Load the data
df, ys, ns, mortality_rates = load_data()
# Generate summary statistics
mu, sigma2, recip_size = summary_stats(ns, mortality_rates)
# Fit hyperparameters alpha and beta using empirical bayes
alpha, beta = empirical_bayes(mu, sigma2, recip_size)
# Compute the recovered moments
mu_rec, sigma2_rec = recovered_moments(alpha, beta, ns)
# Report the results
report_results(ns, mu, sigma2, alpha, beta, mu_rec, sigma2_rec)


# *************************************************************************************************
# 2.3 Use the values of  α  and  β  you derived in 2.2 to generate 5000 posterior samples for the 
# kidney cancer rates for each county. Use these samples to generate a posterior mean rate for each county.

def expected_mortality_rate(alpha: float, beta: float, n: int):
    """The expected mortality rate given alpha and beta"""
    p = (5*n) / (5*n + beta)
    r = alpha
    return p * r / (1-p) / n


def draw_mortality_samples(ys: np.ndarray, ns: np.ndarray, alpha: float, beta: float, num_samples: int):
    """Draw samples for mortality rate by county"""
    # Number of counties
    N = len(ys)
    # Pre-allocate storage; column j is the jth sample.  Row i is the mortality in county i.
    mortality_samples = np.zeros((N, num_samples))
    # Draw N thetas, one for each county
    np.random.seed(42)
    # Iterate over counties
    for i in range(N):
        # The posterior is a negative binomial with parameters
        alpha_post = alpha + ys[i]
        beta_post = beta + 5* ns[i]
        # Convert alpha_post and beta_post into shape and scale
        shape = alpha_post
        scale = 1.0 / beta_post
        # Draw samples from the gamma distribution 
        mortality_samples[i,:] = np.random.gamma(shape=shape, scale=scale, size=num_samples)
    return mortality_samples


# Number of samples
num_samples: int = 5000
# Draw the samples
mortality_samples = draw_mortality_samples(ys, ns, alpha, beta, num_samples)
# The posterior mean is 5 times theta because y is distributed as Pois(5*n*theta)
# and E[y] = 5*n*theta --> E[y]/n = 5*theta
post_means = 5*np.mean(mortality_samples, axis=1)


# *************************************************************************************************
# 2.4. Produce a scatter plot of the raw cancer rates (pct mortality) vs the county population size. 
# Highlight the top 300 raw cancer rates in red. 
# Highlight the bottom 300 raw cancer rates in blue. 
# Finally, on the same plot add a scatter plot visualization of the posterior mean cancer rate estimates 
# (pct mortality) vs the county population size, highlight these in green.

def make_plot(mortality_rates, post_means):
    """Plot kidney cancer mortality rate vs. population size"""
    # Plot the mortality rate * 100,000
    mortality_plot = mortality_rates * 100000
    # Generate mask with the lowest and highest 300 rates
    mask_lo = np.argpartition(mortality_rates, 300)[0:300]
    mask_hi = np.argpartition(mortality_rates, -300)[-300:]
    # Plot requested series 
    fig, ax = plt.subplots()
    fig.set_size_inches([16, 8])
    ax.set_title('Kidney Cancer Mortality Rate vs. Population')
    ax.set_xlabel('County Population Size')
    ax.set_ylabel('Mortality Rate (per 100,000')
    ax.set_xscale('log', basex=2)
    ax.set_ylim(0, 80)
    marker_size = 10
    color_fill = 1.0
    ax.scatter(ns, mortality_plot, color='k', label='Rate', s=marker_size, alpha=color_fill)
    ax.scatter(ns[mask_lo], mortality_plot[mask_lo], color='b', label='Low', s=marker_size, alpha=color_fill)
    ax.scatter(ns[mask_hi], mortality_plot[mask_hi], color='r', label='High', s=marker_size, alpha=color_fill)
    ax.scatter(ns, post_means*100000, color='g', label='Post', s=marker_size, alpha=color_fill)
    ax.grid()
    ax.legend()
    plt.show()

# Make the plots
make_plot(mortality_rates, post_means)


def make_scatter(mortality_rates, post_means):
    """Plot posterior mean vs. observed rate"""
    fig, ax = plt.subplots()
    fig.set_size_inches([16, 8])
    ax.set_title('Kidney Cancer Posterior Rate vs. Empirical Rates')
    ax.set_xlabel('Empirical Mortality Rate (per 100,000)')
    ax.set_ylabel('Posteriror Mortality Rate (per 100,000)')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    mortality_plot = mortality_rates * 100000
    post_plot = post_means * 100000 
    ax.scatter(mortality_plot, post_plot, label='Counties', s=9)
    ax.hline(y=mu*100000,label='Avg')
    ax.grid()
    ax.legend()
    plt.show()

make_scatter(mortality_rates, post_means)