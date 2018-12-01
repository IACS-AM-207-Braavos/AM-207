"""
Harvard AM 207
Homework 11
Problem 1
 
Michael S. Emanuel
Fri Nov 30 10:53:40 2018
"""

import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
from am207_utils import load_vartbl, save_vartbl, plot_style
from IPython.display import display
import warnings
from typing import Dict


# *************************************************************************************************
# Load persisted table of variables
fname: str = 'census_income.pickle'
vartbl: Dict = load_vartbl(fname)

# Fix obscure bug when running code in iPython / Spyder
# https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

# Turn off deprecation warning triggered by theano
warnings.simplefilter('ignore')

# Set plot style
plot_style()

# *************************************************************************************************
# Question 1: Crazy Rich Bayesians Don't Need No Educations?
# *************************************************************************************************

# In this problem, you will explore how to recast data, tasks and research questions from a variety of 
# different contexts so that an existing model can be applied for analysis.

# In this problem, you are given data from the 1994 U.S. Census. 
# The data has been processed so that only a subset of the features are present 
# (for full dataset as well as the description see the UCI Machine Learning Repository). 
# You will be investigate the effect of gender on a person's yearly income in the dataset. 
# In particular, we want to know how a person's gender effect 
# the likelihood of their yearly salary being above or below $50k.

# *************************************************************************************************
# 1.1. Read the dataset into a dataframe and aggregate the dataset by organizing 
# the dataframe into seven different categories.
# *************************************************************************************************

# The categories we wish to consider are:
# Some or no high school
# High school
# Some-college or two year academic college degree
# Professional, vocational school
# 4 year college degree
# Masters
# Doctorate

# Note that you might have to combine some of the existing education categories in your dataframe. 
# For each category, we suggest that you only keep track of a count of the number of males and females who make above 
# (and resp. below) the crazy rich income of $50k (see the dataset in Example 10.1.3).

def load_data():
    """Load data and aggregate it by education and sex"""

    # Read in the full dataframe
    df_full = pd.read_csv('census_data.csv', index_col=0)
    
    # Map from census education categories to new categories
    education_map = {
        'Preschool' : 'Not-HS-Grad',
        '1st-4th' : 'Not-HS-Grad',
        '5th-6th' : 'Not-HS-Grad',
        '7th-8th' : 'Not-HS-Grad',
        '9th' : 'Not-HS-Grad',
        '10th' : 'Not-HS-Grad',
        '11th' : 'Not-HS-Grad',
        '12th' : 'Not-HS-Grad',
        'HS-grad': 'HS-Grad',
        'Some-college': 'College-Lite',
        'Assoc-acdm': 'College-Lite',
        'Assoc-voc': 'Prof-Voc',
        'Prof-school': 'Prof-Voc',
        'Bachelors': 'Bachelors',
        'Masters': 'Masters',
        'Doctorate': 'Doctorate'
        }
    
    educationID_map = {
        'Not-HS-Grad': 0,
        'HS-Grad': 1,
        'College-Lite': 2,
        'Prof-Voc': 3,
        'Bachelors': 4,
        'Masters': 5,
        'Doctorate': 6
    }
    
    sexID_map = {
            'Male' : 0,
            'Female' : 1
    }
    
    # Map from census income to a float (1.0 for high earning)
    earning_map =  {
        '<=50K': 0.0,
        '>50K': 1.0}
    
    # New series for education and earnings
    education = pd.Series(df_full.edu.map(education_map), name='education')
    education_id = pd.Series(education.map(educationID_map), name='education_id', dtype='category')
    # We also want the sex
    sex = df_full.sex
    sex_id = pd.Series(sex.map(sexID_map), name='sex_id', dtype='category')
    # The earnings (trying to predict this)
    earn_hi = pd.Series(df_full.earning.map(earning_map), name='earn_hi', dtype=np.int32)
    
    # Create a new dataframe
    df = pd.concat([education_id, sex_id, education, sex, earn_hi], axis=1, 
                   names=['education_id', 'sex_id', 'education', 'sex', 'earn_hi'])    
    # Aggregate counts of high and low earners by education for males and females as per hint
    # For each category, we suggest that you only keep track of a count of the number of males and females 
    # who make above (and resp. below) the crazy rich income of $50k (see the dataset in Example 10.1.3).
    # https://stackoverflow.com/questions/19384532/how-to-count-number-of-rows-per-group-and-other-statistics-in-pandas-group-by
    gb = df.groupby(by=[education_id, sex_id])
    counts = gb.size().to_frame(name='count')
    df_agg = counts.join(gb.agg({'earn_hi': 'sum'})).reset_index()
    # Change count column to 32 bit integer for compatibility with pymc3 sampling
    df_agg['count'] = df_agg['count'].astype(np.int32)
    # Add indicators is_male, is_female
    df_agg['is_male'] = np.zeros_like(df_agg.education_id, dtype=float)
    df_agg['is_female'] = np.zeros_like(df_agg.education_id, dtype=float)
    # High earning rate in each category
    df_agg['earn_hi_rate'] = df_agg['earn_hi'] / df_agg['count']
    # Return the dataframe    
    return df_agg, educationID_map, sexID_map


def group_by_education(df_agg):
    """Group frames with one row per education level."""
    # Extract male and female
    mask_male = (df_agg.sex_id == 0)
    mask_female = (df_agg.sex_id == 1)
    # Frames for male and female earnings by education
    df_male = df_agg[mask_male]
    df_female = df_agg[mask_female]
    # Columns desired in merge
    columns_merge = ['education_id', 'count_male', 'earnings_male', 'count_female', 'earnings_female']
    df_merged = pd.merge(left=df_male, right=df_female, 
                         on='education_id', suffixes=['_male', '_female'])[columns_merge]
    # df_merged['earn_lo_male'] = df_merged.count_male - df_merged.earnings_male
    df_merged['earn_hi_male'] = df_merged.earnings_male
    # df_merged['earn_lo_female'] = df_merged.count_female - df_merged.earnings_female
    df_merged['earn_hi_female'] = df_merged.earnings_female
    df_merged = df_merged.drop(columns=['earnings_male', 'earnings_female'])
    # Total across male and female
    df_merged['count_person'] = df_merged.count_male + df_merged.count_female
    df_merged['earn_hi_person'] = df_merged.earn_hi_male + df_merged.earn_hi_female
    # Return the aggregated dataframe and 
    return df_merged

# Build the aggregated data frame keyed by (education_id, sex_id)
df, educationID_map, sexID_map = load_data()
display(df)

# *************************************************************************************************
# 1.2. Following Example 10.1.3, build two models for the classification of an individual's yearly income 
# (1 being above $50k and 0 being below), 
# one of these models should include the effect of gender while the other should not.

# Shared configuration for all models in 1.2 & 1.4
# Number of educational categories
num_sex: int = len(sexID_map)
num_edu: int = len(educationID_map)

# Size for models
num_obs: int = len(df)

# Mean and Standard Deviation of distribution for alpha (constant term)
alpha_mu: float = 0.0
alpha_sd: float = 10.0

# Mean and Standard Deviation of distribution for beta (impact of sex on likelihood of high earnings)
beta_sex_mu: float = 0.0
beta_sex_sd: float = 10.0

# Mean and Standard Deviation of distribution for alpha by eductation
beta_edu_mu: float = 0.0
beta_edu_sd: float = 10.0

# The number of samples to draw
num_samples: int = 100000

# *************************************************************************************************
# Create a baseline model with just a constant; name it model_base
with pm.Model() as model_base:
    # The alpha shared by all categories
    alpha = pm.Normal(name='alpha', mu=alpha_mu, sd=alpha_sd)
    # The logit for each category
    logit_p = pm.Deterministic('logit_p', alpha)
    # The probability follows logit(p_i) ~ alpha_i --> p_i ~ invlogit(alpha_i)
    p = pm.Deterministic('p', pm.math.invlogit(logit_p))
    # Data likelihood
    obs_earn = pm.Binomial('obs_earn', n=df['count'].values, p=p, observed=df['earn_hi'].values, dtype='int64')

# Draw samples from model_base
try:
    trace_base = vartbl['trace_base']
    print(f'Loaded trace_base from variable table in {fname}.')
except:    
    with model_base:
        # Need to manually specify cores=1 or this blows up on windows.
        # this is a a known bug on pymc3
        # https://github.com/pymc-devs/pymc3/issues/3140
        trace_base = pm.sample(10000, chains=2, cores=1)
    vartbl['trace_base'] = trace_base
    save_vartbl(vartbl, fname)

# *************************************************************************************************
# Create a model using only sex; name it model_sex
with pm.Model() as model_sex:
    # The beta for the two sex categories
    beta_sex = pm.Normal(name='beta_sex', mu=beta_sex_mu, sd=beta_sex_sd, shape=num_sex)
    # The logit for each category
    logit_p = pm.Deterministic('logit_p', beta_sex[df.sex_id])
    # The probability follows logit(p_i) ~ alpha_i --> p_i ~ invlogit(alpha_i)
    p = pm.Deterministic('p', pm.math.invlogit(logit_p))
    # Data likelihood
    obs_earn = pm.Binomial('obs_earn', n=df['count'].values, p=p, observed=df['earn_hi'].values)

# Draw samples from model_sex
try:
    trace_sex = vartbl['trace_sex']
    print(f'Loaded trace_sex from variable table in {fname}.')
except:    
    with model_sex:
        trace_sex = pm.sample(10000, chains=2, cores=1)
    vartbl['trace_sex'] = trace_sex
    save_vartbl(vartbl, fname)


# *************************************************************************************************
# 1.3. Replicate the analysis in 10.1.3 using your models; specifically, compute wAIC scores and 
# make a plot like Figure 10.5 (posterior check) to see how well your models fits the data.
# *************************************************************************************************
def plot_posterior(HER_data, HER_mean, HER_lo, HER_hi, model_name):
    """Generate the posterior validation plot following the example in Statistical Rethinking"""
    fig, ax = plt.subplots(figsize=[16,8])
    ax.set_title(f'Posterior Validation Check for {model_name} Model')
    # x axis for plots
    xx = np.arange(num_obs)
    ax.set_xticks(xx)
    ax.set_xlabel('Case')
    ax.set_ylabel('High Earning Rate (Above $50k)')

    # Actual data
    p1 = ax.plot(xx, HER_data, marker='o', color='b', markersize=8, linewidth=0, label='Data')
    # Lines between consecutive male / female pairs
    for i in range(num_obs // 2):
        i0 = 2*i
        i1 = i0+2
        ax.plot(xx[i0:i1], HER_data[i0:i1], marker=None, color='b')
    # Mean, Lo, and Hi model estimates
    p2 = ax.plot(xx, HER_lo, marker='_', color='k', markersize=10, linewidth=0, label='Low')
    p3 = ax.plot(xx, HER_mean, marker='o', color='r', markerfacecolor='None', markersize=10, linewidth=0, label='Mean')
    p4 = ax.plot(xx, HER_hi, marker='_', color='k', markersize=10, linewidth=0, label='High')
    # Vertical lines closing up whiskers
    for i in range(num_obs):
        ax.plot(np.array([i,i]), np.array([HER_lo[i], HER_hi[i]]), marker=None, color='k')

    # Legend
    handles = [p1[0], p2[0], p3[0], p4[0]]
    labels = ['Data', 'Low', 'Mean', 'High']
    ax.legend(handles, labels)
    ax.grid()
    plt.show()

# *************************************************************************************************
# Compute WAIC for both models
waic_base = pm.waic(trace_base, model_base)
waic_sex = pm.waic(trace_sex, model_sex)
# Set model names
model_base.name = 'base'
model_sex.name = 'sex'
# Comparison of WAIC
comp_WAIC_base_v_sex = pm.compare({model_base: trace_base, model_sex: trace_sex})
display(comp_WAIC_base_v_sex)
pm.compareplot(comp_WAIC_base_v_sex)

# Generate the posterior predictive in both base and sex models
try:
    post_pred_base = vartbl['post_pred_base']
    post_pred_sex = vartbl['post_pred_sex']
except:
    with model_base:
        post_pred_base = pm.sample_ppc(trace_base)
    with model_sex:
        post_pred_sex = pm.sample_ppc(trace_sex)
    vartbl['post_pred_base'] = post_pred_base
    vartbl['post_pred_sex'] = post_pred_sex
    save_vartbl(vartbl, fname)

# True rate of high earners in each class
HER_data = df['earn_hi_rate'].values

# Mean, low (5.5th percentile), and high (94.5th percentile) estimates of high earning rate (HER) in base model
HER_mean_base = np.mean(post_pred_base['obs_earn'], axis=0) / df['count'].values
HER_lo_base = np.percentile(a=post_pred_base['obs_earn'],q=5.5, axis=0) / df['count'].values
HER_hi_base = np.percentile(a=post_pred_base['obs_earn'],q=94.5, axis=0) / df['count'].values

# HER in sex model
HER_mean_sex = np.mean(post_pred_sex['obs_earn'], axis=0) / df['count'].values
HER_lo_sex = np.percentile(a=post_pred_sex['obs_earn'],q=5.5, axis=0) / df['count'].values
HER_hi_sex = np.percentile(a=post_pred_sex['obs_earn'],q=94.5, axis=0) / df['count'].values

# Plot the posterior validation check for the base model
plot_posterior(HER_data, HER_mean_base, HER_lo_base, HER_hi_base, 'Base')
plot_posterior(HER_data, HER_mean_sex, HER_lo_sex, HER_hi_sex, 'Sex')

# *************************************************************************************************
# 1.4. Following Example 10.1.3, build two models for the classification of an individual's yearly income 
# taking into account education. One of the models should take into account education only 
# the other should take into account gender and education on income.
# *************************************************************************************************

# Create a model using only education; name it model_edu
with pm.Model() as model_edu:
    # The beta for each of the seven educational categories
    beta_edu = pm.Normal(name='beta_edu', mu=beta_edu_mu, sd=beta_edu_sd, shape=num_edu)
    # The logit for each category
    logit_p = pm.Deterministic('logit_p', beta_edu[df.education_id])
    # The probability follows logit(p_i) ~ alpha_i --> p_i ~ invlogit(alpha_i)
    p = pm.Deterministic('p', pm.math.invlogit(logit_p))
    # Data likelihood
    obs_earn = pm.Binomial('obs_earn', n=df['count'].values, p=p, observed=df['earn_hi'].values)

# Draw samples from model_edu
try:
    trace_edu = vartbl['trace_edu']
    print(f'Loaded trace_edu from variable table in {fname}.')
except:    
    with model_edu:
        trace_edu = pm.sample(10000, chains=2, cores=1)
    vartbl['trace_edu'] = trace_edu
    save_vartbl(vartbl, fname)

# Create a model using both education and sex; name it model_edu_sex
with pm.Model() as model_edu_sex:
    # The beta for each of the seven educational categories
    beta_edu = pm.Normal(name='beta_edu', mu=beta_edu_mu, sd=beta_edu_sd, shape=num_edu)
    # The beta for the two sex categories
    beta_sex = pm.Normal(name='beta_sex', mu=beta_sex_mu, sd=beta_sex_sd, shape=num_sex)
    # The logit for each category
    logit_p = pm.Deterministic('logit_p', beta_edu[df.education_id] + beta_sex[df.sex_id])
    # The probability follows logit(p_i) ~ alpha_i --> p_i ~ invlogit(alpha_i)
    p = pm.Deterministic('p', pm.math.invlogit(logit_p))
    # Data likelihood
    obs_earn = pm.Binomial('obs_earn', n=df['count'].values, p=p, observed=df['earn_hi'].values)

# Draw samples from model_edu_sex
try:
    trace_edu_sex = vartbl['trace_edu_sex']
    print(f'Loaded trace_edu from variable table in {fname}.')
except:    
    with model_edu_sex:
        model_edu_sex_trace = pm.sample(10000, chains=2, cores=1)
    vartbl['trace_edu_sex'] = trace_edu_sex
    save_vartbl(vartbl, fname)

# *************************************************************************************************
# 1.5. Replicate the analysis in 10.1.3 using your models; specifically, compute wAIC scores and 
# make a plot like Figure 10.6 (posterior check) to see how well your model fits the data.
# *************************************************************************************************
# Compute WAIC for both models
waic_edu = pm.waic(trace_edu, model_edu)
waic_edu_sex = pm.waic(trace_edu_sex, model_edu_sex)
# Set model names
model_base.name = 'edu'
model_sex.name = 'edu_sex'
# Comparison of WAIC
comp_WAIC_edu_v_both = pm.compare({model_edu: trace_edu, model_edu_sex: trace_edu_sex})
display(comp_WAIC_edu_v_both)
pm.compareplot(comp_WAIC_edu_v_both)

# Generate the posterior predictive in both base and sex models
try:
    post_pred_edu = vartbl['post_pred_edu']
    post_pred_edu_sex = vartbl['post_pred_edu_sex']
except:
    with model_edu:
        post_pred_edu = pm.sample_ppc(trace_edu)
    with model_edu_sex:
        post_pred_edu_sex = pm.sample_ppc(trace_edu_sex)
    vartbl['post_pred_edu'] = post_pred_edu
    vartbl['post_pred_edu_sex'] = post_pred_edu_sex
    save_vartbl(vartbl, fname)

# True rate of high earners in each class
HER_data = df['earn_hi_rate'].values

# Mean, low (5.5th percentile), and high (94.5th percentile) estimates of high earning rate (HER) in base model
HER_mean_edu = np.mean(post_pred_edu['obs_earn'], axis=0) / df['count'].values
HER_lo_edu = np.percentile(a=post_pred_edu['obs_earn'],q=5.5, axis=0) / df['count'].values
HER_hi_edu = np.percentile(a=post_pred_edu['obs_earn'],q=94.5, axis=0) / df['count'].values

# HER in sex model
HER_mean_edu_sex = np.mean(post_pred_edu_sex['obs_earn'], axis=0) / df['count'].values
HER_lo_edu_sex = np.percentile(a=post_pred_edu_sex['obs_earn'],q=5.5, axis=0) / df['count'].values
HER_hi_edu_sex = np.percentile(a=post_pred_edu_sex['obs_earn'],q=94.5, axis=0) / df['count'].values

# Plot the posterior validation check for the base model
plot_posterior(HER_data, HER_mean_edu, HER_lo_edu, HER_hi_edu, 'Education')
plot_posterior(HER_data, HER_mean_edu_sex, HER_lo_edu_sex, HER_hi_edu_sex, 'Education & Sex')

# *************************************************************************************************
# 1.6. Using your analysis from 1.3, discuss the effect gender has on income.
# *************************************************************************************************



# *************************************************************************************************
# 1.7. Using your analysis from 1.5, discuss the effect of gender on income 
# taking into account an individual's education.
# *************************************************************************************************

