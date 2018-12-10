"""
Harvard Applied Math 207
Final Exam
Problem 1

Michael S. Emanuel
Sat Dec  8 10:05:42 2018
"""

# Core calculations
import numpy as np
import scipy.stats
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from theano.printing import pydotprint
# Charting & display
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from IPython.display import display
# Miscellaneous
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

# Turn off deprecation warning (too noisy)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=mpl.MatplotlibDeprecationWarning)

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
mpl.rcParams.update({'font.size': 16})
fig = plt.figure(figsize=(12,20))
gs = pm.forestplot(trace_fe, ylabels=[f'dist {i}' for i in range(num_districts)],)
gs.figure = fig
ax1, ax2 = fig.axes
ax1.set_xlim(0.8, 1.2)
plt.close(fig)

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

# Summary of the variable-effects model
summary_ve = pm.summary(trace_ve)

# Samples of alpha as a an Nx60 array
alpha_overall_ve = trace_ve.get_values('alpha')
alpha_district_samples_ve = trace_ve.get_values('alpha_district')
alpha_samples_ve = np.hstack([alpha_overall_ve.reshape(-1,1), alpha_district_samples_ve])

# Arrange the alpha samples into a dataframe for plotting
col_names_ve = ['alpha'] + [f'alpha_{i}' for i in range(num_districts)]
df_alpha_samples_ve = pd.DataFrame(data=alpha_samples_ve, columns = col_names_ve)

# Samples of alpha as an Nx60 array
alpha_overall_ve = trace_ve.get_values('alpha')
alpha_district_samples_ve = trace_ve.get_values('alpha_district')
alpha_samples_ve = np.hstack([alpha_overall_ve.reshape(-1,1), alpha_district_samples_ve]).shape
y_labels = ['intercept'] + [f'district {i}' for i in range(num_districts)]

# Arrange the alpha samples into a dataframe for plotting
col_names_ve = ['alpha'] + [f'alpha_{i}' for i in range(num_districts)]
df_alpha_samples_ve = pd.DataFrame(data=alpha_samples_fe, columns = col_names_fe)

# *************************************************************************************************
# A3 What does a posterior-predictive sample in this model look like? 
# What is the difference between district specific posterior predictives and woman specific posterior predictives. 
# In other words, how might you model the posterior predictive for a new woman being from a particular district vs 
# that of a new woman in the entire sample? 
# This is a word answer; no programming required.

# See notebook

# *************************************************************************************************
# A4 Plot the predicted proportions of women in each district using contraception 
# against the id of the district, in both models. 
# How do these models disagree? Look at the extreme values of predicted contraceptive use in the fixed effects model. 
# How is the disagreement in these cases?

# Generate posterior predictive samples in both models
num_samples_ppc: int = 10000
try:
    post_pred_fe = vartbl['post_pred_fe']
    post_pred_ve = vartbl['post_pred_ve']
    print(f'Loaded posterior predictive samples for Fixed Effects and Varying Effects models.')
except:
    post_pred_fe = pm.sample_ppc(trace_fe, samples=num_samples_ppc, model=model_fe)
    post_pred_ve = pm.sample_ppc(trace_ve, samples=num_samples_ppc, model=model_ve)
    vartbl['post_pred_fe'] = post_pred_fe
    vartbl['post_pred_ve'] = post_pred_ve

# Compute the mean contraception use in posterior samples of each model
df['use_contraception_fe'] = np.mean(post_pred_fe['use_contraception'], axis=0)
df['use_contraception_ve'] = np.mean(post_pred_ve['use_contraception'], axis=0)

# Update the aggregated contraception use in each district
agg_tbl = {
        'woman': ['count'],
        'use_contraception': ['mean'],
        'use_contraception_fe': ['mean'],
        'use_contraception_ve': ['mean'],        
        } 
df_district = df.groupby(by=df.district_id).agg(agg_tbl)
df_district.columns = ["_".join(x) for x in df_district.columns.ravel()]
# Change column names to make model suffix at the end of the name
df_district.rename(axis='columns', inplace=True, mapper=
    {'use_contraception_fe_mean':'use_contraception_mean_fe',
     'use_contraception_ve_mean':'use_contraception_mean_ve',
     })

# Set up horizontal bar chart
district_id_agg = df_district.index.values
# Spacing between models in each district
space = 0.3
plot_y1 = district_id_agg - space
plot_y2 = district_id_agg
plot_y3 = district_id_agg + space
# Height of each horizontal bar
height = 0.2

# Plot contraception use for each district
fig, ax = plt.subplots(figsize=[16,20])
ax.set_title('Contraception Use vs. DistrictID by Model')
ax.set_xlabel('Contraception Use in District')
ax.set_ylabel('District ID')
ax.set_ylim(0, 60)
ax.barh(y=plot_y1, width=df_district.use_contraception_mean, height=height, label='Data', color='r')
ax.barh(y=plot_y2, width=df_district.use_contraception_mean_fe, height=height, label='FE', color='g')
ax.barh(y=plot_y3, width=df_district.use_contraception_mean_ve, height=height, label='VE', color='b')
ax.legend()
ax.grid()
plt.close(fig)

# *************************************************************************************************
# A5 Plot the absolute value of the difference in probability of contraceptive use against the number 
# of women sampled in each district. What do you see?

# Assemble series for this plot
# the x-axis is the number of women in the district
plot_x = df_district.woman_count.values
# the y-axis is the absolute value of the difference between the FE and VE models
plot_y = np.abs(df_district.use_contraception_mean_fe - df_district.use_contraception_mean_ve).values

# Generate the plot
fig, ax = plt.subplots(figsize=[12,8])
ax.set_title('Difference Between FE and VE Models vs. Sample Size')
ax.set_xlabel('Sample Size (# Women Polled)')
ax.set_ylabel('Abs(pred_FE - pred_VE)')
ax.plot(plot_x, plot_y, color='b', marker='o', markersize=8, linewidth=0)
ax.grid()
plt.close(fig)

# *************************************************************************************************
# Part B.
# Let us now fit a model with both varying intercepts by district_id (like we did in the varying effects model above) 
# and varying slopes of urban by district_id. 
# *************************************************************************************************

# To do this, we will
# (a) have an overall intercept, call it alpha
# (b) have an overall slope of urban, call it beta.
# (c) have district specific intercepts alpha_district
# (d) district specific slopes for urban, beta_district
# (e) model the co-relation between these slopes and intercepts.
# We have not modelled covariance and correlation before, 
# so look at http://am207.info/wiki/corr.html for notes on how this is done.

# To see the ideas behind this, see section 13.2.2 on the income 
# data from your textbook (included as a pdf in this zip). 
# Feel free to use code with attribution from Osvaldo Martin..with attribution and understanding...
# there is some sweet pymc3 technical wrangling in there.


# *************************************************************************************************
# B1 Write down the model as a pymc3 specification and look at its graph. 
# Note that this model builds a 60 by 2 matrix with alpha_district values in the first column 
# and beta_district values in the second. By assumption, the first column and the second column 
# have correlation structure given by an LKJ prior, but there is no explicit correlation among the rows. 
# In other words, the correlation matrix is 2x2 (not 60x60). 
# Make sure to obtain the value of the off-diagonal correlation as a pm.Deterministic. 
# (See Osvaldo Martin's code above)
# *************************************************************************************************

def pm_make_cov(sigma_priors, corr_coeffs, ndim):
    """Assemble a covariance matrix single variable standard deviations and correlation coefficients"""
    # Citation: AM 207 lecture notes: http://am207.info/wiki/corr.html
    # Diagonal matrix of standard deviation for each varialbes
    sigma_matrix = tt.nlinalg.diag(sigma_priors)
    # A symmetric nxn matrix has n choose 2 = n(n-1)/2 distinct elements
    n_elem = int(ndim * (ndim - 1) / 2)
    # Convert between array indexing and [i, j) indexing
    tri_index = np.zeros([ndim, ndim], dtype=int)
    tri_index[np.triu_indices(ndim, k=1)] = np.arange(n_elem)
    tri_index[np.triu_indices(ndim, k=1)[::-1]] = np.arange(n_elem)
    # Assemble the covariance matrix using the equation
    # CovMat = DiagMat * CorrMat * DiagMat
    corr_matrix = corr_coeffs[tri_index]
    corr_matrix = tt.fill_diagonal(corr_matrix, 1)
    return tt.nlinalg.matrix_dot(sigma_matrix, corr_matrix, sigma_matrix)

# Define a varying slopes model incorporating a beta_urban term
with pm.Model() as model_vs:  
    # Set the prior for the overall intercept
    alpha = pm.Normal(name='alpha', mu=0.0, sd=10.0)
    # Set the prior for the overall intercept on urban, beta
    beta = pm.Normal(name='beta', mu=0.0, sd=10.0)
    
    # Citation: http://am207.info/wiki/corr.html for code controlling correlation structure
    # The parameter nu is the prior on correlation; 0 is uniform, infinity is no corelation
    nu = pm.Uniform('nu', 1.0, 5.0)
    # The number of dimensions here is 2: correlation structure is bewteen alpha and beta by district
    num_factors: int = 2
    # Sample the correlation coefficients using the LKJ distribution
    corr_coeffs = pm.LKJCorr('corr_coeffs', nu, num_factors)

    # Sample the variances of the single factors
    sigma_priors = tt.stack([pm.Lognormal('sigma_prior_alpha', mu=0.0, tau=1.0),
                             pm.Lognormal('sigma_prior_beta', mu=0.0, tau=1.0)])

    # Make the covariance matrix as a Theano tensor
    cov = pm.Deterministic('cov', pm_make_cov(sigma_priors, corr_coeffs, num_factors))
    # The multivariate Gaussian of (alpha, beta) by district
    theta_district = pm.MvNormal('theta_district', mu=[0.0, 0.0], cov=cov, shape=(num_districts, num_factors))   

    # The vector of standard deviations for each variable; size num_factors x num_factors
    # Citation: efficient generation of sigmas and rhos from cov
    # https://github.com/aloctavodia/Statistical-Rethinking-with-Python-and-PyMC3/blob/master/Chp_13.ipynb
    sigmas = pm.Deterministic('sigmas', tt.sqrt(tt.diag(cov)))
    # correlation matrix (num_factors x num_factors)
    rhos = pm.Deterministic('rhos', tt.diag(sigmas**-1).dot(cov.dot(tt.diag(sigmas**-1))))

    # Extract the standard deviations of alpha and beta, and the correlation coefficient rho
    sigma_alpha = pm.Deterministic('sigma_alpha', sigmas[0])
    sigma_beta = pm.Deterministic('sigma_beta', sigmas[1])
    rho = pm.Deterministic('rho', rhos[0, 1])

    # Extract alpha_district and beta_district from theta_district
    alpha_district = pm.Deterministic('alpha_district', theta_district[:,0])
    beta_district = pm.Deterministic('beta_district', theta_district[:, 1])

    # Set the probability that each woman uses contraception in this model
    # It depends on the district she lives in and whether the district is urban
    # p = pm.math.invlogit(alpha + alpha_district[df.district_id] + 
    #                      (beta + beta_district[df.district_id]) * df.urban)
    p = pm.math.invlogit(alpha + theta_district[df.district_id, 0] + 
                         (beta + theta_district[df.district_id, 1]) * df.urban)

    # The response variable - whether this woman used contraception; modeled as Bernoulli
    # Bind this to the observed values
    use_contraception = pm.Bernoulli('use_contraception', p=p, observed=df['use_contraception'])

# Generate graphs for each model
graph_fe = pm.model_to_graphviz(model_ve)
graph_ve = pm.model_to_graphviz(model_ve)
graph_vs = pm.model_to_graphviz(model_vs)

# *************************************************************************************************
# B2: Sample from the posterior of the model above with a target acceptance rate of .9 or more. 
# (Sampling takes me 7 minutes 30 seconds on my 2013 Macbook Air). 
# Comment on the quality of the samples obtained.
# *************************************************************************************************

# Sample from the varying-slope model
try:
    trace_vs = vartbl['trace_vs']
    print(f'Loaded samples for the Variable Slopes model in trace_vs.')
except:
    with model_vs:
        nuts_kwargs = {'target_accept': 0.90}
        trace_vs = pm.sample(draws=num_samples, tune=num_tune, nuts_kwargs=nuts_kwargs, chains=2, cores=1)
    vartbl['trace_vs'] = trace_vs
    save_vartbl(vartbl, fname)

# Summary of the variable-effects model
summary_vs = pm.summary(trace_vs)

# *************************************************************************************************
# B3 Propose a method based on the reparametrization trick for multi-variate gaussians) 
# of improving the quality of the samples obtained and implement it. 
# (A hint can be obtained from here: 
# https://docs.pymc.io/api/distributions/multivariate.html#pymc3.distributions.multivariate.MvNormal . 
# Using that hint lowered the sampling time to 2.5 minutes on my laptop).
# *************************************************************************************************

# Define a varying slopes model incorporating a beta_urban term
with pm.Model() as model_vsr:
    # Citation: ideas to efficiently reparameterize samples from a MV Gaussian
    # https://docs.pymc.io/api/distributions/multivariate.html#pymc3.distributions.multivariate.MvNormal
    # Set the prior for the overall intercept
    alpha = pm.Normal(name='alpha', mu=0.0, sd=10.0)
    # Set the prior for the overall intercept on urban, beta
    beta = pm.Normal(name='beta', mu=0.0, sd=10.0)
    
    # Sample the variances of the single factors
    # sd_dist = pm.HalfCauchy.dist(beta=2.5, shape=num_factors)
    sd_dist = pm.Lognormal.dist(mu=0.0, tau=1.0, shape=num_factors)
    # The parameter nu is the prior on correlation; 0 is uniform, infinity is no corelation
    eta = pm.Uniform('nu', 1.0, 5.0)
    # The number of dimensions here is 2: correlation structure is bewteen alpha and beta by district
    num_factors: int = 2
    # Sample the correlation coefficients using the LKJ distribution
    chol_packed = pm.LKJCholeskyCov('chol_packed', n=num_factors, eta=eta, sd_dist = sd_dist)
    # Expand the packed Cholesky matrix to full size
    chol = pm.Deterministic('chol', pm.expand_packed_triangular(num_factors, chol_packed))
    # Make the covariance matrix by multiplying out the cholesky factor by its transpose
    cov = pm.Deterministic('cov', tt.dot(chol, chol.T))
    # The multivariate Gaussian of (alpha, beta) by district
    # Decompose this into a "raw" part and then scale it
    theta_raw = pm.Normal(name='theta_raw', mu=0.0, sd=1.0, shape=(num_districts, num_factors))   
    # Now scale these to have the desired covariance structure
    theta_district = pm.Deterministic(name='theta_district', var=tt.dot(chol, theta_raw.T).T)
    
    # The vector of standard deviations for each variable; size num_factors x num_factors
    # Citation: efficient generation of sigmas and rhos from cov
    # https://github.com/aloctavodia/Statistical-Rethinking-with-Python-and-PyMC3/blob/master/Chp_13.ipynb
    sigmas = pm.Deterministic('sigmas', tt.sqrt(tt.diag(cov)))
    # correlation matrix (num_factors x num_factors)
    rhos = pm.Deterministic('rhos', tt.diag(sigmas**-1).dot(cov.dot(tt.diag(sigmas**-1))))

    # Extract the standard deviations of alpha and beta, and the correlation coefficient rho
    sigma_alpha = pm.Deterministic('sigma_alpha', sigmas[0])
    sigma_beta = pm.Deterministic('sigma_beta', sigmas[1])
    rho = pm.Deterministic('rho', rhos[0, 1])

    # Extract alpha_district and beta_district from theta_district
    alpha_district = pm.Deterministic('alpha_district', theta_district[:,0])
    beta_district = pm.Deterministic('beta_district', theta_district[:, 1])

    # Set the probability that each woman uses contraception in this model
    # It depends on the district she lives in and whether the district is urban
    # p = pm.math.invlogit(alpha + alpha_district[df.district_id] + 
    #                      (beta + beta_district[df.district_id]) * df.urban)
    p = pm.math.invlogit(alpha + theta_district[df.district_id, 0] + 
                         (beta + theta_district[df.district_id, 1]) * df.urban)

    # The response variable - whether this woman used contraception; modeled as Bernoulli
    # Bind this to the observed values
    use_contraception = pm.Bernoulli('use_contraception', p=p, observed=df['use_contraception'])

# Sample from the reparameterized varying-slope model
try:
    trace_vsr = vartbl['trace_vsr']
    print(f'Loaded samples for the Variable Slopes Reparameterized model in trace_vsr.')
except:
    with model_vsr:
        nuts_kwargs = {'target_accept': 0.90}
        trace_vsr = pm.sample(draws=num_samples, tune=num_tune, nuts_kwargs=nuts_kwargs, chains=2, cores=1)
    vartbl['trace_vsr'] = trace_vsr
    save_vartbl(vartbl, fname)

# Summary of the variable-effects model
summary_vsr = pm.summary(trace_vsr)

# *************************************************************************************************
# B4 Inspect the trace of the correlation between the intercepts and slopes, plotting the correlation marginal.
# What does this correlation tell you about the pattern of contraceptive use in the sample? 
# It might help to plot the mean (or median) varying effect estimates for both the intercepts and slopes, by district. 
# Then you can visualize the correlation and maybe more easily think through 
# what it means to have a particular correlation. 
# Also plot the predicted proportion of women using contraception, with urban women on one axis and rural on the other. 
# Finally, also plot the difference between urban and rural probabilities against rural probabilities. 
# All of these will help you interpret your findings. 
# (Hint: think in terms of low or high rural contraceptive use)
# *************************************************************************************************

# Plot the mean varying effect estimates for both the intercepts and slopes, by district
# List of parameter names for alpha_district and beta_district for each district
district_suffix = [f'district__{i}' for i in range(num_districts)]
params_alpha_district = [f'alpha_{suffix}' for suffix in district_suffix]
params_beta_district = [f'beta_{suffix}' for suffix in district_suffix]

# Get the mean of alpha_district and beta_district over all the districts
alpha_district_mean = summary_vsr.loc[params_alpha_district]['mean'].values
beta_district_mean = summary_vsr.loc[params_beta_district]['mean'].values
# Mean of "global" parameters alpha, beta, and rho
alpha_mean = summary_vsr.loc['alpha']['mean']
beta_mean = summary_vsr.loc['beta']['mean']
rho_mean = summary_vsr.loc['rho']['mean']

# Plot beta vs. alpha
fig, ax = plt.subplots(figsize=[12,8])
ax.set_title('Mean Beta vs. Mean Alpha By District')
ax.set_xlabel('Mean alpha_district for Each District over Samples')
ax.set_ylabel('Mean beta_district for Each District over Samples')
ax.plot(alpha_district_mean, beta_district_mean, label='data', color='b', linewidth=0, marker='o', markersize=8)
ax.plot(alpha_district_mean, alpha_district_mean * rho_mean, label=r'$\rho \alpha$', linewidth=4, color='r')
ax.legend()
ax.grid()


# *************************************************************************************************
# B5 Add additional "slope" terms (one-by-one) into the model for
# (a) the centered-age of the women and
# (b) an indicator for whether the women have a small number or large number of existing kids in the house (you can treat 1-2 kids as low, 3-4 as high, but you might want to experiment with this split).
# Are any of these effects significant? Are any significant effects similar over the urban/rural divide?
# *************************************************************************************************


# *************************************************************************************************
# B6 Use WAIC to compare your models. What are your conclusions?
# *************************************************************************************************
