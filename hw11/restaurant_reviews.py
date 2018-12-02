"""
Harvard Applied Math 207
Homework 11
Problem 3

Michael S. Emanuel
Sun Dec  2 10:18:12 2018
"""

import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.special import erf
import warnings
from am207_utils import load_vartbl, save_vartbl
from typing import Dict


# *************************************************************************************************
# Load persisted table of variables
fname: str = 'restaurant_reviews.pickle'
vartbl: Dict = load_vartbl(fname)

# Fix obscure bug when running code in iPython / Spyder
# https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

# Turn off deprecation warning triggered by theano
warnings.simplefilter('ignore')

# Set small font size for plots
mpl.rcParams.update({'font.size': 12})


# *************************************************************************************************
# Use 1-cdf at 0.5 to model the probability of having positive sentiment
# it basically tells you the area under the gaussian after 0.5 (we'll assume 
# positive sentiment based on the usual probability > 0.5 criterion)

prob = lambda mu, vari: .5 * (1 - erf((0.5- mu) / np.sqrt(2 * vari)))

# fix a restaurant and an aspect (food or service)
# "means" is the array of values in the "mean" column for the restaurant and the aspect 
#         in the dataset
# "thetas" is the array of values representing your estimate of the opinions of reviewers 
#          regarding this aspect of this particular restaurant
# "theta_vars" is the array of values of the varaiances of the thetas
# "counts" is the array of values in the "count" column for the restaurant and the aspect 
#.         in the dataset
# FEEL FREE TO RE-IMPLEMENT THESE

def shrinkage_plot(means, thetas, mean_vars, theta_vars, counts, ax):
    """
    a plot that shows how review means (plotted at y=0) shrink to
    review $theta$s, plotted at y=1
    """
    data = zip(means, thetas, mean_vars / counts, theta_vars, counts)   
    palette = itertools.cycle(sns.color_palette())
    with sns.axes_style('white'):
        for m,t, me, te, c in data: # mean, theta, mean errir, thetax error, count
            color=next(palette)
            # add some jitter to y values to separate them
            noise=0.04*np.random.randn()
            noise2=0.04*np.random.randn()
            if me==0:
                me = 4
            # plot shrinkage line from mean, 0 to
            # theta, 1. Also plot error bars
            ax.plot([m,t],[noise,1+noise2],'o-', color=color, lw=1)
            ax.errorbar([m,t],[noise,1+noise2], xerr=[np.sqrt(me), np.sqrt(te)], color=color,  lw=1)
        ax.set_yticks([])
        ax.set_xlim([0,1])
        sns.despine(offset=-2, trim=True, left=True)
    return plt.gca()


def prob_shrinkage_plot(means, thetas, mean_vars, theta_vars, counts, ax):
    """
    a plot that shows how review means (plotted at y=prob(mean)) shrink to
    review $theta$s, plotted at y=prob(theta)
    """
    data = zip(means, thetas, mean_vars / counts, theta_vars, counts)
    palette = itertools.cycle(sns.color_palette())
    with sns.axes_style('white'):
        for m,t, me, te, c in data: # mean, theta, mean errir, theta error, count
            color = next(palette)
            # add some jitter to y values to separate them
            noise = 0.001 * np.random.randn()
            noise2 = 0.001 * np.random.randn()
            if me == 0: #make mean error super large if estimated as 0 due to count=1
                me = 4
            p = prob(m, me)
            peb = prob(t, te)
            # plot shrinkage line from mean, prob-based_on-mean to
            # theta, prob-based_on-theta. Also plot error bars
            ax.plot([m, t],[p, peb],'o-', color=color, lw=1)
            ax.errorbar([m, t],[p + noise, peb + noise2], xerr=[np.sqrt(me), np.sqrt(te)], color=color, lw=1)
        ax = plt.gca()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
    return ax

# *************************************************************************************************
# 3.1. Following the SAT prep school example discussed in lab (and influenced your answers for HW 10 Question #1), 
# set up a Bayesian model (that is, write functions encapsulating the pymc3 code) for a reviewer  
# j's opinion of restaurant k's food and service (considering the food and service separately). 
# You should have a model for each restaurant and each aspect being reviewed (food and serivce). 
# For restaurant k, you will have a model for {θ_food_jk} and one for {θ_service_jk}, 
# where θ_jk is the positivity of the opinion of the j-th reviewer regarding the k-th restaurant.
# *************************************************************************************************

# Hint: What quantity in our data naturally corresponds to  y¯j 's in the prep school example? 
# How would you calculate the parameter  σ2j  in the distribution of  y¯j 
# (note that, contrary to the school example,  σ2j  is not provided explictly in the restaurant data)?

# Load the review data    
reviews_df = pd.read_csv('reviews_processed.csv')

def get_rest_topic_model(reviews_df, topic, rid):
    """Return a model describing the given topic and restaurant with the restaurant ID = rid."""
    rest_topic = reviews_df.loc[(reviews_df.rid == rid) & (reviews_df['count'] > 1) & (reviews_df.topic == topic)]
    j_obs = rest_topic.shape[0]
    y_obs = rest_topic['mean'].values
    sigma_obs = np.sqrt(rest_topic['var'].values / rest_topic['count'].values)
    
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=.5, sd=.15)
        tau = pm.HalfCauchy('tau', beta=.1)
        nu = pm.Normal('nu', mu=0, sd=.5, shape=j_obs)
        theta = pm.Deterministic('theta', mu + tau*nu)
        obs = pm.Normal('obs', mu=theta, sd=sigma_obs, observed=y_obs)
        
    return model


def get_rest_model(data, rid):
    """
    Return a tuple of two models describing the food (topic=0) and service (topic=1) 
    of the restuarant with restaurant id = rid."""
    return get_rest_topic_model(data, 0, rid), get_rest_topic_model(data, 1, rid)


# *************************************************************************************************
# 3.2 Just to test your that modeling makes sense choose 1 restaurant and run your model from 3.1 
# on the food and service aspects for that restaurant. 
# Create 10K samples each for the food and service model for your chosen restuarant and 
# visualize your samples via a traceplot for each aspect of the restaurant reviews.
# *************************************************************************************************
    
# Get the restaurant ID for the review in slot 6 in reviews_df
rid = reviews_df.rid[6]
# Get the models for food and service on this restaurant
food_model, service_model = get_rest_model(reviews_df, rid)

# Draw 10,000 samples from the food model
try:
    food_trace = vartbl['food_trace']
    print(f'Loaded food_trace from {fname}.')
except:
    with food_model:
        # Need to manually specify cores=1 to avoid broken pipes bug on Windows platform
        food_trace = pm.sample(draws=10000, init=None, tune=1000, cores=1)
    vartbl['food_trace'] = food_trace
    save_vartbl(vartbl, fname)

# Draw 10,000 samples from the service model
try:
    service_trace = vartbl['service_trace']
    print(f'Loaded service_trace from {fname}.')
except:
    with service_model:
        # Need to manually specify cores=1 to avoid broken pipes bug on Windows platform
        service_trace = pm.sample(draws=10000, init=None, tune=1000, cores=1)
    vartbl['service_trace'] = service_trace
    save_vartbl(vartbl, fname)

def plot_traces():
    # Display the traceplot for food reviews on this restaurant
    print('Food:')
    pm.traceplot(food_trace)
    plt.show()
    
    # Display the traceplot for food reviews on this restaurant
    print('Service:')
    pm.traceplot(service_trace)
    plt.show()

# *************************************************************************************************
# 3.3. Use your model from 3.1 to produce estimates for θjk 's for multiple restaurants. 
# Pick a few (try for 5 but if computer power is a problem, choose 2) restaurants and for each aspect 
# ("food" and "service") of each restaurant, plot your estimates for the θ 's against 
# the values in the "mean" column (corresponding to this restaurant).
# *************************************************************************************************

def get_ths(trace):
    """Get the mean theta for each restaurant"""
    return trace['theta'].mean(axis=0).tolist()


def get_th_vars(trace):
    """Get the variance of theta for each restaurant"""
    return trace['theta'].var(axis=0).tolist()

# Build an array with all the distinct restaurants that are reviewed
all_rids = np.array(list(set(reviews_df.rid)))
# Generate a sample with 5 of them
rests = all_rids[[5,6,7,8,9]].tolist()
rest_count = len(rests)

# Build models for each restaurant (on both food and service)
models = [get_rest_model(reviews_df, rid) for rid in rests]

# Sample traces for the food and service models of each retaurant
try:
    traces = vartbl['traces']
    print(f'Loaded traces for {rest_count} restaurants from {fname}.')
except:
    traces = []
    for food_mod, service_mod in models:
        with food_mod:
            food_trace = pm.sample(draws=10000, init=None, tune=1000, cores=1)
        with service_mod:
            service_trace = pm.sample(draws=10000, init=None, tune=1000, cores=1)
        traces.append((food_trace, service_trace))
    vartbl['traces'] = traces
    save_vartbl(vartbl, fname)

# *************************************************************************************************
# Compute the mean for the food and service reviews for each restaurant sampled
means_food, means_service = [], []
for rid in rests:
    food_idx = (reviews_df.rid == rid) & (reviews_df['count'] > 1) & (reviews_df.topic == 0)
    means_food.append(reviews_df.loc[food_idx]['mean'].values)
    service_idx = (reviews_df.rid == rid) & (reviews_df['count'] > 1) & (reviews_df.topic == 1)
    means_service.append(reviews_df.loc[service_idx]['mean'].values)

# Extract the estimated thetas for each resturant
ths_food, ths_service = [], []
for food_trace, service_trace in traces:
    ths_food.append(get_ths(food_trace))
    ths_service.append(get_ths(service_trace))

# Plot the food and service means for each restaurant
mpl.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(ncols=2, figsize=(18, 5))
sns.stripplot(data=means_food, color=sns.color_palette()[0], ax=axes[0], label='Means')
sns.stripplot(data=ths_food, color=sns.color_palette()[1], ax=axes[0], label=r'$\theta$s')
sns.stripplot(data=means_service, color=sns.color_palette()[0], ax=axes[1], label='Means')
sns.stripplot(data=ths_service, color=sns.color_palette()[1], ax=axes[1], label=r'$\theta$s')
axes[0].set_title(r'Food Means vs. $\theta$s', fontsize=16)
axes[1].set_title(r'Service Means vs. $\theta$s', fontsize=16)
for ax in axes:
    ax.set_xlabel('Restaurant')
    ax.set_ylabel('Value')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[0], handles[-1]], [labels[0], labels[-1]], loc='lower left')
plt.show()


# *************************************************************************************************
# Compute the variance for the food and service reviews for each restaurant 
# (this is a variance over the actual data)
mean_vars_food, mean_vars_service = [], []
for rid in rests:
    food_idx = (reviews_df.rid == rid) & (reviews_df['count'] > 1) & (reviews_df.topic == 0)
    mean_vars_food.append(reviews_df.loc[food_idx]['var'].values)
    service_idx = (reviews_df.rid == rid) & (reviews_df['count'] > 1) & (reviews_df.topic == 1)
    mean_vars_service.append(reviews_df.loc[service_idx]['var'].values)

# Count the number of food and service reviews for each restaurant sampled
counts_food, counts_service = [], []
for rid in rests:
    food_idx = (reviews_df.rid == rid) & (reviews_df['count'] > 1) & (reviews_df.topic == 0)
    counts_food.append(reviews_df.loc[food_idx]['count'].values)
    service_idx = (reviews_df.rid == rid) & (reviews_df['count'] > 1) & (reviews_df.topic == 1)
    counts_service.append(reviews_df.loc[service_idx]['count'].values)

# Extract the estimated variance of the thetas for each each restaurant
# (this is a variance over sampled paramaters data)
th_vars_food, th_vars_service = [], []
for food_trace, service_trace in traces:
    th_vars_food.append(get_th_vars(food_trace))
    th_vars_service.append(get_th_vars(service_trace))

# Generate the shrinkage plots for food and service
mpl.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(19, 20))
for iax, ax in enumerate(axes.ravel()):
    topic = iax % 2
    r = iax // 2
    if topic:
        shrinkage_plot(means_service[r], ths_service[r], mean_vars_service[r], th_vars_service[r], counts_service[r], ax)
        ax.set_title(f'Restaurant {r} Service Shrinkage Plot')
    else:
        shrinkage_plot(means_food[r], ths_food[r], mean_vars_food[r], th_vars_food[r], counts_food[r], ax)
        ax.set_title(f'Restaurant {r} Food Shrinkage Plot')

# Generate the probability shrinkage plots for food and service
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(19, 30))
for iax, ax in enumerate(axes.ravel()):
    topic = iax % 2
    r = iax // 2
    if topic:
        prob_shrinkage_plot(means_service[r], ths_service[r], mean_vars_service[r], th_vars_service[r], counts_service[r], ax)
        ax.set_title(f'Restaurant {r} Service Prob Shrinkage Plot')
    else:
        prob_shrinkage_plot(means_food[r], ths_food[r], mean_vars_food[r], th_vars_food[r], counts_food[r], ax)
        ax.set_title(f'Restaurant {r} Food Prob Shrinkage Plot')

# *************************************************************************************************
# 3.4. Based on your shrinkage plots and probability shrinkage plots in 3.3 discuss the statistical benefits 
# of modeling each reviewer's opinion using your hierarchical model rather than approximating 
# the reviewer opinion with the value in "mean".
# *************************************************************************************************


# *************************************************************************************************
# 3.5. Aggregate, in a simple but reasonable way, the reviewer's opinions given a pair of overall scores 
# for each restaurant -- one for food and one for service. 
# Rank the restaurants by food score and then by service score.
# (Hint: Think what an average score for each aspect would do here?)
# *************************************************************************************************


# *************************************************************************************************
# 3.6. Discuss the statistical weakness of ranking by these scores.
# (Hint: What is statistically problematic about the way you aggregated the reviews of each restaurant to 
# produce an overall food or service score? This is also the same problem with summarizing a reviewer's 
# opinion on a restaurants service and food based on what they write.)
# *************************************************************************************************
