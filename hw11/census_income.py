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
from am207_utils import load_vartbl, save_vartbl
from IPython.display import display
from typing import Dict


# *************************************************************************************************
# Load persisted table of variables
fname: str = 'census_income.pickle'
vartbl: Dict = load_vartbl(fname)

# Fix obscure bug when running code in iPython / Spyder
# https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

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
    earn_hi = pd.Series(df_full.earning.map(earning_map), name='earn_hi', dtype=np.int64)
    
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
    # Add indicators is_male, is_female
    df_agg['is_male'] = np.zeros_like(df_agg.education_id, dtype=float)
    df_agg['is_female'] = np.zeros_like(df_agg.education_id, dtype=float)
    # Set is_male to 1.0 on the male entries
    mask_male = (df_agg.sex_id == 0)
    df_agg.loc[mask_male, 'is_male'] = 1.0
    # Set is_female to 1.0 on the female entries
    mask_female = (df_agg.sex_id == 1)
    df_agg.loc[mask_female, 'is_female'] = 1.0
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
    obs_earn = pm.Binomial('obs_earn', n=df['count'].values, p=p, observed=df['earn_hi'].values)

# Draw samples from model_base
try:
    model_base_trace = vartbl['model_base_trace']
    print(f'Loaded model_base_trace from variable table in {fname}.')
except:    
    with model_base:
        # Need to manually specify cores=1 or this blows up on windows.
        # this is a a known bug on pymc3
        # https://github.com/pymc-devs/pymc3/issues/3140
        model_base_trace = pm.sample(10000, chains=2, cores=1)
    vartbl['model_base_trace'] = model_base_trace
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
    model_sex_trace = vartbl['model_sex_trace']
    print(f'Loaded model_sex_trace from variable table in {fname}.')
except:    
    with model_sex:
        # Need to manually specify cores=1 or this blows up on windows.
        # this is a a known bug on pymc3
        # https://github.com/pymc-devs/pymc3/issues/3140
        model_sex_trace = pm.sample(10000, chains=2, cores=1)
    vartbl['model_sex_trace'] = model_sex_trace
    save_vartbl(vartbl, fname)


# *************************************************************************************************
# 1.3. Replicate the analysis in 10.1.3 using your models; specifically, compute wAIC scores and 
# make a plot like Figure 10.5 (posterior check) to see how well your models fits the data.
# *************************************************************************************************


# *************************************************************************************************
# 1.4. Following Example 10.1.3, build two models for the classification of an individual's yearly income 
# taking into account education. One of the models should take into account education only 
# the other should take into account gender and education on income.
# *************************************************************************************************

# Create a model using only education; name it model_edu
with pm.Model() as model_edu:
    # The alpha (baseline) shared by all the educational categories
    alpha = pm.Normal(name='alpha', mu=alpha_mu, sd=alpha_sd)
    # The beta for each of the seven educational categories
    beta_edu = pm.Normal(name='beta_edu', mu=beta_edu_mu, sd=beta_edu_sd, shape=num_obs)
    # The probability follows logit(p_i) ~ alpha_i --> p_i ~ invlogit(alpha_i)
    p = pm.Deterministic('p', pm.math.invlogit(beta_edu + alpha))
    # Data likelihood
    obs_earn = pm.Binomial('obs_earn', n=df['count'].values, p=p, observed=df['earn_hi'].values)

# Draw samples from model_edu
try:
    model_edu_trace = vartbl['model_edu_trace']
    print(f'Loaded model_edu_trace from {fname}.')
except:    
    with model_edu:
        # Need to manually specify cores=1 or this blows up on windows.
        # this is a a known bug on pymc3
        # https://github.com/pymc-devs/pymc3/issues/3140
        model_edu_trace = pm.sample(10000, chains=2, cores=1)
    vartbl['model_edu_trace'] = model_edu_trace
    save_vartbl(vartbl, fname)


# Create a model using both education and sex; name it model_edu_sex
with pm.Model() as model_edu_sex:
    # The mean for each of the seven educational categories
    alpha = pm.Normal(name='alpha', mu=alpha_mu, sd=alpha_sd, shape=num_obs)
    # The impact of sex; 
    beta = pm.Normal(name='beta', mu=beta_mu, sd=beta_sd)
    # Array with values of the dummy variable is_male
    is_male = df.is_male.values    
    # The probability follows logit(p_i) ~ alpha_i + beta * is_male_i --> p_i ~ invlogit(alpha_i + beta * is_male_i)
    p = pm.Deterministic('p', pm.math.invlogit(alpha + beta * is_male))
    # Data likelihood
    obs_earn = pm.Binomial('obs_earn', n=df['count'].values, p=p, observed=df['earn_hi'].values)

# Draw samples from model_edu
try:
    model_edu_sex_trace = vartbl['model_edu_trace']
    print(f'Loaded model_edu_sex_trace from {fname}.')
except:    
    with model_edu_sex:
        model_edu_sex_trace = pm.sample(10000, chains=2, cores=1)
    vartbl['model_edu_sex_trace'] = model_edu_sex_trace
    save_vartbl(vartbl, fname)



# *************************************************************************************************
# 1.5. Replicate the analysis in 10.1.3 using your models; specifically, compute wAIC scores and 
# make a plot like Figure 10.6 (posterior check) to see how well your model fits the data.
# *************************************************************************************************


# *************************************************************************************************
# 1.6. Using your analysis from 1.3, discuss the effect gender has on income.
# *************************************************************************************************



# *************************************************************************************************
# 1.7. Using your analysis from 1.5, discuss the effect of gender on income 
# taking into account an individual's education.
# *************************************************************************************************

