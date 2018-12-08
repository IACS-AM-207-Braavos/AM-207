"""
Harvard Applied Math 207
Final Exam
Problem 1

Michael S. Emanuel
Sat Dec  8 10:05:42 2018
"""

import numpy as np
import scipy.stats
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from IPython.display import display
import warnings
from am207_utils import load_vartbl, save_vartbl
from typing import Dict

# *************************************************************************************************
# Load persisted table of variables
fname: str = 'glm.pickle'
vartbl: Dict = load_vartbl(fname)

# Fix obscure bug when running code in iPython / Spyder
# https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

# Silence warnings (too noisy)
# warnings.simplefilter('ignore')

# *************************************************************************************************
# Q1: GLMs with correlation
# The dataset: A Bangladesh Contraception use census
# *************************************************************************************************

# This problem is based on one-two (12H1 and continuations ) from your textbook.
# The data is in the file bangladesh.csv. These data are from the 1988 Bangladesh Fertility Survey. 
# Each row is one of 1934 women. There are six variables:

# (1) district: ID number of administrative district each woman resided in
# (2) use.contraception: An indicator (0/1) of whether the woman was using contraception
# (3) urban: An indicator (0/1) of whether the woman lived in a city, as opposed to living in a rural area
# (4) woman: a number indexing a single woman in this survey
# (5) living.chidren: the number of children living with a woman
# (6) age.centered: a continuous variable representing the age of the woman with the sample mean subtracted

# We need to make sure that the cluster variable, district, is a contiguous set of integers, 
# so that we can use the index to differentiate the districts easily while sampling 
# ((look at the Chimpanzee models we did in lab to understand the indexing). 
# So create a new contiguous integer index to represent the districts. 
# Give it a new column in the dataframe, such as district.id.

# You will be investigating the dependence of contraception use on the district in which the survey was done. 
# Specifically, we will want to regularize estimates from those districts where very few women were surveyed. 
# We will further want to investigate whether the areas of residence 
# (urban or rural) within a district impacts a woman's use of contraception.

# Feel free to indulge in any exploratory visualization which helps you understand the dataset better.

# *************************************************************************************************
# Load the dataset
df = pd.read_csv('bangladesh.csv', sep=';')
# Generate map of distinct districts
district_id_map = {d:i for i, d in enumerate(sorted(set(df.district)))}
# Add new column district_id to the dataset; these are contiguous integers 0-59
df['district_id'] = df.district.map(district_id_map)

# Rename the columns to avoid the confusing '.' in column names
df.rename(axis='columns', inplace=True, mapper=
    {'use.contraception':'use_contraception',
     'living.children':'living_children',
     'age.centered':'age_centered',
     })

# The number of districts
num_districts: int = len(district_id_map)

# Create a dataset aggregated by district (sufficient statistic for the by district model)
agg_tbl = {
        'woman': ['count'],
        'use_contraception': ['sum']
        } 
df_district = df.groupby(by=df.district_id).agg(agg_tbl)
df_district.columns = ["_".join(x) for x in df_district.columns.ravel()]

# Set the number of samples for this problem (used in multiple parts)
num_samples: int = 11000
num_tune: int = 1000

# Part A
# We will use use.contraception as a Bernoulli response variable.
# When we say "fit" below, we mean, specify the model, plot its graph, sample from it, 
# do some tests, and forest-plot and summarize the posteriors, at the very least.

# *************************************************************************************************
# A1 Fit a traditional "fixed-effects" model which sets up district-specific intercepts, 
# each with its own Normal(0, 10) prior. That is, the intercept is modeled something like
# alpha_district = pm.Normal('alpha_district', 0, 10, shape=num_districts)
# p=pm.math.invlogit(alpha_district[df.district_id])
# Why should there not be any overall intercept in this model?

# Define the fixed-effects model
with pm.Model() as model_fe:
    # Set the prior for the intercept in each district
    alpha_district = pm.Normal(name='alpha_district', mu=0.0, sd=10.0, shape=num_districts)
    # Set the probability that each woman uses contraception in this model
    # It depends only on the district she lives in
    p = pm.math.invlogit(alpha_district[df.district_id])
    # The response variable - whether this woman used contraception; modeled as Bernoulli
    # Bind this to the observed values
    use_contraception = pm.Bernoulli('use_contraception', p=p, observed=df['use_contraception'])

# Sample from the fixed-effects model
try:
    trace_fe = vartbl['trace_fe']
    print(f'Loaded samples for the Fixed Effects model in trace_fe.')
except:
    with model_fe:
        trace_fe = pm.sample(draws=num_samples, tune=num_tune, chains=2, cores=1)
    vartbl['trace_fe'] = trace_fe
    save_vartbl(vartbl, fname)

# Summary of the fixed-effects model
summary_fe = pm.summary(trace_fe)
# Samples of alpha as a an Nx60 array
alpha_samples_fe = trace_fe.get_values('alpha_district')
# Arrange the alpha samples into a dataframe for plotting
col_names_fe = [f'alpha_{i}' for i in range(num_districts)]
df_alpha_samples_fe = pd.DataFrame(data=alpha_samples_fe, columns = col_names_fe)


# forest plot for district model
fig = plt.figure(figsize=(12,20))
gs = pm.forestplot(trace_fe, ylabels=[f'dist {i}' for i in range(num_districts)],)
gs.figure = fig
ax1, ax2 = fig.axes
ax1.set_xlim(0.8, 1.2)

# *************************************************************************************************
# A2 Fit a multi-level "varying-effects" model with an overall intercept alpha, 
# and district-specific intercepts alpha_district. 
# Assume that the overall intercept has a Normal(0, 10) prior, while the district specific intercepts 
# are all drawn from the same normal distribution with mean 0 and standard deviation σ. 
# Let σ be drawn from HalfCauchy(2). 
# The setup of this model is similar to the per-chimanzee models in the prosocial chimanzee labs.

# Define the varying-effects model
with pm.Model() as model_ve:
    # Set the prior for the overall intercept
    alpha = pm.Normal(name='alpha', mu=0.0, sd=10.0)
    # Set the width sigma for the variability among districts
    sigma = pm.HalfCauchy(name='sigma', beta=2.0)
    # Set the district-specific alphas to have mean 0 and standard deviation sigma
    alpha_district = pm.Normal(name='alpha_district', mu=0.0, sd=sigma, shape=num_districts)    
    # Set the probability that each woman uses contraception in this model
    # It depends only on the district she lives in
    p = pm.math.invlogit(alpha + alpha_district[df.district_id])
    # The response variable - whether this woman used contraception; modeled as Bernoulli
    # Bind this to the observed values
    use_contraception = pm.Bernoulli('use_contraception', p=p, observed=df['use_contraception'])

# Sample from the variable-effects model
try:
    trace_ve = vartbl['trace_ve']
    print(f'Loaded samples for the Variable Effects model in trace_ve.')
except:
    with model_ve:
        trace_ve = pm.sample(draws=num_samples, tune=num_tune, chains=2, cores=1)
    vartbl['trace_ve'] = trace_ve
    save_vartbl(vartbl, fname)

# Summary of the fixed-effects model
summary_ve = pm.summary(trace_ve)
# Samples of alpha as a an Nx60 array
alpha_samples_ve = trace_ve.get_values('alpha_district')
# Arrange the alpha samples into a dataframe for plotting
col_names_ve = ['alpha'] + [f'alpha_{i}' for i in range(num_districts)]
df_alpha_samples_ve = pd.DataFrame(data=alpha_samples_ve, columns = col_names_ve)


# *************************************************************************************************
# A3 What does a posterior-predictive sample in this model look like? 
# What is the difference between district specific posterior predictives and woman specific posterior predictives. 
# In other words, how might you model the posterior predictive for a new woman being from a particular district vs 
# that os a new woman in the entire sample? 
# This is a word answer; no programming required.


# *************************************************************************************************
# A4 Plot the predicted proportions of women in each district using contraception against the id of the district, 
# in both models. 
# How do these models disagree? Look at the extreme values of predicted contraceptive use in the fixed effects model. 
# How is the disagreement in these cases?


# *************************************************************************************************
# A5 Plot the absolute value of the difference in probability of contraceptive use against the number 
# of women sampled in each district. What do you see?
