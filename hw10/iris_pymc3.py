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



# Run the traceplot
pm.traceplot(model_trace, 'beta_const')