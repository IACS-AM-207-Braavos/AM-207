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
from scipy.special import gamma, binom

# Load the data
df = pd.read_csv('kcancer.csv')

# Death count: y[j]
ys = df['dc'].values
# Population of each county: n[j]
ns = df['pop'].values
# Mortality rate of each county
mortality_rates = df['pct_mortality']

# Summary statistics
N = int(np.sum(ys))

# Sample mean and variance over counties
# Use correct weights for mean rate (i.e. pool)
mu = np.average(mortality_rates, weights=ns)
# Sample variance over counties - this time equally weight the counties
sigma2 = np.var(mortality_rates)
sigma = sqrt(sigma2)

# Pooled variance
np.average((mortality_rates-mu)**2, weights=ns)

# Mean of reciprocal size
recip_size = np.mean(1.0 / ns)
recip_size_pooled = np.average(1.0/ns, weights=ns)

# Implied parameters alpha and beta
beta = 5.0 / (sigma2 / mu - recip_size)
alpha = mu * beta / 5.0
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
sigma_rec = sqrt(sigma2_rec)



# Report results
mortality_rate = mu
print(f'Total number of people: N={N}.')
print(f'Overall cancer mortality rate:     {mortality_rate:0.6f}')
print(f'Overall cancer mortality approximately one in {int(round(1/mortality_rate))}')
print(f'Population mean mu = {mu:0.6f}')
print(f'Population variance sigma2 = {sigma2:0.3e}')
print(f'Population std dev sigma   = {sigma:0.6f}')

print(f'\nEstimated Model Parameters for Gamma prior by moment matching:')
print(f'alpha={alpha:0.6f}')
print(f'beta={beta:0.6f}')
print(f'Recovered mu = {mu_rec:0.6f}')
print(f'Recovered sigma2 = {sigma2_rec:0.3e}')
print(f'Recovered sigma = {sigma_rec:0.6f}')


def prob_y(y: int, alpha: float, beta: float, n: int):
    """Probability of y cancer mortalities in a population of N"""
    p: float = (5*n) / (5*n+beta)
    q: float = (beta) / (5*n + beta)
    # The binomial term
    bt: float = binom(y + alpha - 1, alpha)
    return bt * p**y * q**alpha


