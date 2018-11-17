"""
Harvard Applied MAth 207
Homework 10
Problem 2

Michael S. Emanuel
Fri Nov 16 20:19:21 2018
"""

import numpy as np
import pymc3 as pm
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from am207_utils import load_vartbl, save_vartbl, plot_style
from typing import Dict

plot_style()

# *************************************************************************************************
# Load persisted table of variables
fname: str = 'iris_pymc3.pickle'
vartbl: Dict = load_vartbl(fname)


# *************************************************************************************************
# Question 2: In a Flash the Iris devient un Fleur-de-Lis
# *************************************************************************************************

# We've done classification before, but the goal of this problem is to introduce you 
# to the idea of classification using Bayesian inference.
# Consider the famous Fisher flower Iris data set a multivariate data set introduced 
# by Sir Ronald Fisher (1936) as an example of discriminant analysis. 
# The data set consists of 50 samples from each of three species of Iris 
# (Iris setosa, Iris virginica and Iris versicolor). 
# Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. 
# Based on the combination of these four features, you will build a model to predict the species.
# For this problem only consider two classes: virginica and not-virginica.

# Let (X,Y) be our dataset, where  X={x_1,…x_n} and x_i is the standard feature vector corresponding to an offset 1 
# and the four components explained above.  Y∈{0,1} are the scalar labels of a class. 
# In other words the species labels are your Y data (virginica = 0 and virginica=1), 
# and the four features -- petal length, petal width, sepal length and sepal width -- 
# along with the offset make up your  X  data.

# The goal is to train a classifier, that will predict an unknown class label ŷ from a new data point x .

# Consider the following glm (logistic model) for the probability of a class:
# p(y)=1 / (1+exp(−xTβ))


# *************************************************************************************************
# 2.1. Use a 60-40 stratified (preserving class membership) split of the dataset into a training set and a test set. 
# (Feel free to take advantage of scikit-learn's train_test_split).

def get_mats(df):
    """Get trainable matrices from the dataframe"""
    # Extract features as an Nx4 matrix
    X_feat = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    # Shape of features
    m, n = X_feat.shape
    # Augment features with a constant column in the first slot (index 0)
    X = np.ones((m, n+1))
    X[:,1:n+1] = X_feat
    # Get class labels
    y_names = df['class'].values
    # Get rid of leading spaces in class labels
    y_names = [y_name.strip() for y_name in y_names]
    # Convert labels to virginica=1, not virginica=0
    y = np.array([y_name == 'Iris-virginica' for y_name in y_names]).astype(np.float)
    # Split into train-test
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
    return X_trn, X_tst, y_trn, y_tst

# Load the data
df = pd.read_csv('iris.csv')
# Get trainable matrices
X_trn, X_tst, y_trn, y_tst = get_mats(df)


# *************************************************************************************************
# 2.2. Choose a prior for  β∼N(0,σ2I)  and write down the formula for the posterior p(β|Y,X) . 
# Since we dont care about regularization here, just use the mostly uninformative value  σ=10 .
# *************************************************************************************************

# In notebook


# *************************************************************************************************
# 2.3. Find the MAP for the posterior on the training set.
# *************************************************************************************************

with pm.Model() as model:
    # See helpful Q&A on Piazza @362 :HW10 Q2.3 Some tips for building your PYMC model
    # http://barnesanalytics.com/bayesian-logistic-regression-in-python-using-pymc3
    
    # Prior on beta
    # Number of predictors INCLUDING the constant
    n=5
    # Prior of the variance entries
    sigma2 = 10.0
    # Use a mean of zero
    mu = np.zeros(n)
    # Use a diagonal covariance matrix
    cov = sigma2 * np.identity(n)
    # Beta is a multivariate normal
    beta = pm.MvNormal('beta', mu=mu, cov=cov, shape=5)

    # Probability each data point has y=1 given X and beta: must use pm.math.dot, not numpy.dot!
    prob_pos = pm.math.sigmoid(pm.math.dot(X_trn, beta))

    # Likelihood is a Bernoulli process; success probability computed above, observed values in training set
    likilihood = pm.Bernoulli('y', p=prob_pos, observed=y_trn)

    # Name each beta coefficient so we have them later for traceplot
    beta_0 = pm.Deterministic('beta_0', beta[0])
    beta_1 = pm.Deterministic('beta_1', beta[1])
    beta_2 = pm.Deterministic('beta_2', beta[2])
    beta_3 = pm.Deterministic('beta_3', beta[3])
    beta_4 = pm.Deterministic('beta_4', beta[4])


# Find the MAP estimate of beta on the training data
# Hard code the asnwer from a previous run as the STARTING POINT ONLY
# This is just to save time on future runs of the program!
try:    
    beta_MAP = vartbl['beta_MAP']
except:
    beta_MAP = np.array([-3.96061023, -2.12694628, -2.81210817,  3.58328363,  4.60231749])
    param_start = {'beta':beta_MAP}
    param_MAP = pm.find_MAP(start=param_start, model=model)
    beta_MAP = param_MAP['beta']
    vartbl['beta_MAP'] = beta_MAP
    save_vartbl(vartbl, fname)


# *************************************************************************************************
# 2.4 Implement a PyMC3 model to sample from this posterior of  β .
# *************************************************************************************************


# *************************************************************************************************
# 2.5. Generate 5000 samples of β. 
# Visualize the betas and generate a traceplot and autocorrelation plots for each beta component.
# *************************************************************************************************

# Generate the samples
try:
    # Only draw samples of beta if they are not already known
    model_trace = vartbl['model_trace']
except:
    with model:
        stepper = pm.NUTS()
        # Fix obscure bug when running code in iPython / Spyder
        # https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
        __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
        # Draw the samples; increase from default number of tuning samples b/c PyMC3 was complaining
        model_trace = pm.sample(draws=5000, stepper=stepper, tune=1000, cores=16)
        # Save the model trace to the variable table
        vartbl['model_trace'] = model_trace
        save_vartbl(vartbl, fname)

# Extract the sampled values of beta from the model trace
beta_samples = model_trace['beta']
# Feature names for plots
beta_names = ['Const', 'Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']


# Visualize the betas
fig, axs = plt.subplots(5)
fig.suptitle(r'Histograms of $\beta_i$')
fig.set_size_inches([16, 16])
plt.subplots_adjust(hspace=0.4)
for i, ax in enumerate(axs):
    beta_name = beta_names[i]
    ax.set_title(f'$beta_{i}$ for {beta_name}'.replace(r'$beta', r'$\beta'))
    ax.hist(beta_samples[:,i], bins=100)
    ax.grid()



# Run the traceplot for beta as a composite
pm.traceplot(model_trace, varnames=['beta'])

# Run the traceplot for beta_i's individually
var_names = ['beta_0', 'beta_1', 'beta_2', 'beta_3', 'beta_4']
for i in range(n):
    pm.traceplot(model_trace, varnames=[var_names[i]])
    plt.suptitle(beta_names[i])

# Run a small trace with just one chain so the autocorrelation plots are legible
try:
    model_trace_small = vartbl['model_trace_small']
except:    
    with model:
        stepper = pm.NUTS()
        model_trace_small = pm.sample(draws=5000, stepper=stepper, tune=1000, chains=1, cores=16)
    vartbl['model_trace_small'] = model_trace_small
    
# Run the autocorrelation plots for beta_i's individually
for i in range(n):
    pm.autocorrplot(model_trace_small, varnames=[var_names[i]])


# *************************************************************************************************
# 2.6 Based on your samples construct an estimate for the posterior mean.
# *************************************************************************************************

beta_PM = np.mean(beta_samples, axis=0)
print(f'Estimated Posterior Means:')
for i, beta_name in enumerate(beta_names):
    print(f'{beta_name:13} = {beta_PM[i]:+0.3f}')
    
    
# *************************************************************************************************
# 2.7. Select at least 2 datapoints and visualize a histogram of the posterior probabilities. 
# Denote the posterior mean and MAP on your plot for each datapoint
# *************************************************************************************************

def logit(x):
    """Logistic function (sigmoid)"""
    return 1 / (1 + np.exp(-x))


def plot_test(x_tst):
    # Values of z over beta samples
    zs = beta_samples @ x_tst
    # Probabilities (full histogram)
    probs = logit(zs)
    # Probabilities at posterior mean and MAP
    prob_PM = logit(beta_MAP @ x_tst)
    prob_MAP = logit(beta_PM @ x_tst)
    # Plot histogram
    fig, ax = plt.subplots()
    fig.set_size_inches([16, 8])
    ax.set_title('Histogram of Posterior Probabilities of a Test Point')
    ax.set_xlabel('Prob. Point is Virginica')
    ax.set_ylabel('Frequency')
    ax.hist(probs, bins=100, density=True, label='Hist', color='b')
    ax.axvline(x=prob_PM, label=r'$\beta$ PM', color='r', linewidth=4)
    ax.axvline(x=prob_MAP, label=r'$\beta$ MAP', color='g', linewidth=4)
    ax.legend()
    ax.grid()


plot_test(X_tst[0,:])
plot_test(X_tst[1,:])


# *************************************************************************************************
# 2.8 Plot the distributions of  pMEAN ,  pCDF ,  pMAP  and  pPP  over all the data points in the training set. 
# How are these different?
# *************************************************************************************************

# Sample the posterior predictive for p_PP calculation
try:
    post_pred = vartbl['post_pred']
except:
    with model:
        post_pred = pm.sample_ppc(model_trace)
        vartbl['post_pred'] = post_pred
        save_vartbl(vartbl, fname)
# Unpack y from post_pred
y_pred = post_pred['y']

# p_MEAN for each data point is the logit of the dot product of x_tst with beta_PM
p_MEAN = logit(X_trn @ beta_PM)
# p_MAP for each data point is the logit of the dot product of x_tst with beta_MAP
p_MAP = logit(X_trn @ beta_MAP)
# A given test subject / sample combination will be classified as positive when p > 0.5
# Take the mean of this binary condition to efficiently compute p_CDF
p_CDF = np.mean(y_pred > 0.5, axis=0)
# Take the mean of the posterior predictive for each sample point
p_PP = np.mean(y_pred, axis=0)

fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Histograms of Different Probability Predictions')
ax.set_xlabel('Probability of Virginica')
ax.set_ylabel('Frequency')
bins = 100
alpha=0.5
ax.hist(p_MEAN, bins=bins, label='p_MEAN', alpha=alpha)
ax.hist(p_MAP, bins=bins, label='p_MAP', alpha=alpha)
ax.hist(p_CDF, bins=bins, label='p_CDF', alpha=alpha)
ax.hist(p_PP, bins=bins, label='p_PP', alpha=alpha)
ax.legend()
ax.grid()

fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Probability Histogram for p_MEAN')
ax.set_xlabel('Probability of Virginica')
ax.set_ylabel('Frequency')
bins = 100
ax.hist(p_MEAN, bins=bins, label='p_MEAN')
ax.legend()
ax.grid()

fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Probability Histogram for p_MAP')
ax.set_xlabel('Probability of Virginica')
ax.set_ylabel('Frequency')
bins = 100
ax.hist(p_MAP, bins=bins, label='p_MAP')
ax.legend()
ax.grid()

fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Probability Histogram for p_CDF')
ax.set_xlabel('Probability of Virginica')
ax.set_ylabel('Frequency')
bins = 100
ax.hist(p_CDF, bins=bins, label='p_CDF')
ax.legend()
ax.grid()

fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Probability Histogram for p_PP')
ax.set_xlabel('Probability of Virginica')
ax.set_ylabel('Frequency')
bins = 100
ax.hist(p_PP, bins=bins, label='p_PP')
ax.legend()
ax.grid()


# *************************************************************************************************
# 2.9 Plot the posterior-predictive distribution of the misclassification rate with respect to the true 
# class identities y(x) of the data points x (in other words you are plotting a histogram with the 
# misclassification rate for the  ntrace  posterior-predictive samples) on the training set.
# *************************************************************************************************

# Compute the error rate on each of the 5000 predictive samples over 90 training points
error_rate = np.mean((y_pred != y_trn), axis=1)
# Plot error rate on posterior predictive
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Error Rate Histogram for Posterior Predictive Samples')
ax.set_xlabel('Probability of Virginica')
ax.set_ylabel('Frequency')
bins = 50
ax.hist(error_rate, bins=bins)
ax.grid()


# *************************************************************************************************
# 2.10 For every posterior sample, consider whether the data point ought to be classified as a 1 or 0 
# from the p>0.5⟹y=1  decision theoretic prespective. 
# Using the MLDT defined above, overlay a plot of the histogram of the misclassification rate for the posterior 
# on the corresponding plot for the posterior-predictive you constructed in 2.9. 
# Which case (from posterior-predictive or from-posterior) has a wider mis-classification distribution?
# *************************************************************************************************

# Filter beta_samples down from 80,000 to 5,000 to match
beta_samples = beta_samples[0:5000]
# Compute z = beta * x for each training point
z_trn = X_trn @ beta_samples.T
# Compute the ML style predictions: z > 0 implies a positive prediction
y_pred_ml = (z_trn.T > 0).astype(np.float)
# Error rate for this kind of predition
error_rate_ml = np.mean((y_pred_ml != y_trn), axis=1)
# Plot error rate on ML predictions
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Error Rate Histogram for MLDT Samples')
ax.set_xlabel('Probability of Virginica')
ax.set_ylabel('Frequency')
bins = 50
ax.hist(error_rate_ml, bins=bins)
ax.grid()



# *************************************************************************************************
# 2.11 Repeat 2.9 and 2.10 for the test set (i.e. make predictions). 
# Describe and interpret the widths of the resulting distributions.
# *************************************************************************************************

# Sample the posterior predictive on test data
# I am going to do this by hand because it's simple, and shared variables in PyMC3 are complicated.
# Compute z = beta * x for each test point
z_tst = X_tst @ beta_samples.T
# Compute probabilities on the test set
prob_tst = logit(z_tst)
# Sample Bernoulli trials    
z_tst_rand = np.random.uniform(size=z_tst.shape)
y_pred_tst = (z_tst_rand < prob_tst).T.astype(np.float)
# Compute error rate on test data for posterior predictive distribution
error_rate_tst = np.mean((y_pred_tst != y_tst), axis=1)

# Compute the ML style preditions on test data
y_pred_ml_tst = (z_tst.T > 0).astype(np.float)
# Error rate for this kind of predition
error_rate_ml_tst = np.mean((y_pred_ml_tst != y_tst), axis=1)

# Plot error rate on posterior predictive
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Error Rate Histogram for Posterior Predictive Samples on Test')
ax.set_xlabel('Probability of Virginica')
ax.set_ylabel('Frequency')
bins = 50
ax.hist(error_rate_tst, bins=bins)
ax.grid()

# Plot error rate on ML predictions on test
fig, ax = plt.subplots()
fig.set_size_inches([16, 8])
ax.set_title('Error Rate Histogram for Posterior Predictive Samples')
ax.set_xlabel('Probability of Virginica')
ax.set_ylabel('Frequency')
bins = 50
ax.hist(error_rate_ml, bins=bins)
ax.grid()
