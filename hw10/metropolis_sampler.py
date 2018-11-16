"""
Harvard Applied MAth 207
Homework 10

Michael S. Emanuel
Wed Nov 14 10:50:44 2018
"""

import numpy as np
from numpy import exp

# *************************************************************************************************
# 1.1. Implement a Metropolis sampler to produce sample guesses from 500 individuals, 
# with the λ values,  λ=0.2,0.5,1.0 . What are the top five possible guesses?
# *************************************************************************************************

# *************************************************************************************************
def hamming_dist(x: np.ndarray, y: np.ndarray):
    """Compute the Hamming Distance between two arrays"""
    return np.sum(x != y)


def proposal(x: np.ndarray):
    """Generate a proposal guess by transposing a random pair of elements in x."""
    # Initialize a copy of x
    y = x.copy()
    # Draw a random pair in [0, n)
    n: int = len(x)
    i, j = np.random.choice(n, 2, replace=False)
    # Swap the ith and jth elements
    xi, xj = x[i], x[j]
    y[i], y[j] = xj, xi
    # Return modified guess
    return y


def prob(x: np.ndarray, y: np.ndarray, lam: float):
    """The unscaled probability of guess theta given omega and the lambda parameter (inverse temperature)."""
    return exp(-lam * hamming_dist(x, y))


def metropolis(num_samp: int, lam: float, x_init: np.ndarray):
    """Metropolis Sampler for this problem"""
    # The correct order
    x_best: np.ndarray = np.array([1,2,3,4,5], dtype=np.int8)
    # Get the length n of each guess
    n: int = len(x_best)
    # Initialize array to store the samples
    samples: np.ndarray = np.empty((num_samp, n), dtype=np.int8)
    # Initialize x_prev
    x_prev: np.ndarray = x_init
    # Draw num_samp samples using Metrpolis algorithm
    # Adapted from example code in Lecture 16, p. 6
    for i in range(num_samp):
        # The proposed new point
        x_star: np.ndarray = proposal(x_prev)
        # Compare probability of the propsed point to the current one
        p_star: float = prob(x_star, x_best, lam)
        p_prev: float = prob(x_prev, x_best, lam)
        # Probability ratio
        pdf_ratio: float = p_star / p_prev
        # Randomly accept or reject the step based on pdf_ratio
        if np.random.uniform() < min(1.0, pdf_ratio):
            # Accept the step
            samples[i] = x_star
            x_prev = x_star
        else:
            # Reject the step; duplicate x_prev as a sample
            samples[i] = x_prev
    return samples


def top_five_guesses(samples: np.ndarray):
    """Given an Nx5 array of guesses, find the top five."""
    # Count the frequency of the distinct guesses
    guesses, counts = np.unique(samples, return_counts=True, axis=0)
    # Sort the guesses by their frequency
    gc = list(zip(guesses, counts))
    gc.sort(key = lambda x: x[1], reverse=True)
    # Return the top five guesses and their frequency
    gc = gc[0:5]
    guesses = [x[0] for x in gc]
    counts = [x[1] for x in gc]
    return guesses, counts


# *************************************************************************************************
# Set random seed
np.random.seed(42)
# Number of desired samples 
num_samp = 5000
# Set burn-in
burn_in = 100
# Pick a random starting point
x_init = np.random.choice(5, size=5, replace=False).astype(np.int8) + 1
# Range of lambdas to test
lams = (0.2, 0.5, 1.0)
# Table for the results
sample_tbl = dict()
top_five_guesses_tbl = dict()
# Test each lambda in turn
for lam in lams:
    # Run a burn-in period of 100 samples
    discarded_samples = metropolis(burn_in, lam, x_init)
    # Run the metropolis sampler on the current value of x
    x_init = discarded_samples[-1, :]
    samples = metropolis(num_samp, lam, x_init)
    # Save samples in the table
    sample_tbl[lam] = samples
    # Get the top 5 guesses and their frequencies
    guesses, counts = top_five_guesses(samples)
    # Save top five guesses to table
    top_five_guesses_tbl[lam] = top_five_guesses(samples)

# Display results
for lam in lams:
    guesses, counts = top_five_guesses_tbl[lam]
    print(f'\nLambda = {lam:0.1f}.  Top five guesses and frequency:')
    for i in range(5):
        print(f'{guesses[i]}, {counts[i]:3} ({counts[i]/num_samp*100:0.2f})%.')



# *************************************************************************************************
# 1.2. Compute the probability that *The Shawshank Redemption* is ranked as the top movie (ranked number 1) 
# by the Metropolis algorithm sampler. Compare the resulting probabilities for the various λ values.
# *************************************************************************************************

# Set large number of samples
num_samp_big = 50000
# Table for the results
sample_tbl_big = dict()
# Test each lambda in turn
for lam in lams:
    # Run the metropolis sampler on the current value of x
    x_init = samples[-1, :]
    samples = metropolis(num_samp_big, lam, x_init)
    # Save samples in the table
    sample_tbl_big[lam] = samples

# Iterate through the lambdas
print('')
for lam in lams:
    # Extract the samples
    samples = sample_tbl_big[lam]
    # Count how often "Shawshank" is rated first
    shawshank_wins = np.sum(samples[:, 0]==1)
    shawshank_win_prob = shawshank_wins / num_samp_big
    # Report results
    print(f'Lambda = {lam:0.1f}.  Probability Shawshank ranked first = {shawshank_win_prob*100:0.2f}%.')



# *************************************************************************************************
def test_hamming():
    omega = np.array([1,2,3,4,5])
    theta = np.array([2,3,5,4,1])
    d = hamming_dist(omega, theta)
    assert d == 4
    # print(f'Hamming Distance between omega=(1,2,3,4,5) and theta=(2,3,5,4,1) is {d}.')
    print(f'Test Hamming Distance:\n*** PASS ***')

test_hamming()
