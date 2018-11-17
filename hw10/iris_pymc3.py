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
    y = np.array([y_name == 'Iris-virginica' for y_name in y_names])
    # Split into train-test
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
    return X_trn, X_tst, y_trn, y_tst

# Load the data
df = pd.read_csv('iris.csv')
# Get trainable matrices
X_trn, X_tst, y_trn, y_tst = get_mats(df)


