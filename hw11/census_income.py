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
    earnings = pd.Series(df_full.earning.map(earning_map), name='earnings')
    
    # Create a new dataframe
    df = pd.concat([education_id, sex_id, education, sex, earnings], axis=1, 
                   names=['education_id', 'sex_id', 'education', 'sex', 'earning'])
    
    # Aggregate counts of high and low earners by education for males and females as per hint
    # For each category, we suggest that you only keep track of a count of the number of males and females 
    # who make above (and resp. below) the crazy rich income of $50k (see the dataset in Example 10.1.3).
    # https://stackoverflow.com/questions/19384532/how-to-count-number-of-rows-per-group-and-other-statistics-in-pandas-group-by
    gb = df.groupby(by=[education_id, sex_id])
    counts = gb.size().to_frame(name='count')
    df_agg = counts.join(gb.agg({'earnings': 'sum'})).reset_index()
    # Extract male and female
    mask_male = (df_agg.sex_id == 0)
    mask_female = (df_agg.sex_id == 1)
    # Frames for male and female earnings by education
    df_male = df_agg[mask_male]
    df_female = df_agg[mask_female]
    # Columns desired in merge
    columns_merge = ['education_id', 'count_male', 'earnings_male', 'count_female', 'earnings_female']
    df_merged = pd.merge(left=df_male, right=df_female, on='education_id', suffixes=['_male', '_female'])[columns_merge]
    df_merged['earn_lo_male'] = df_merged.count_male - df_merged.earnings_male
    df_merged['earn_hi_male'] = df_merged.earnings_male
    df_merged['earn_lo_female'] = df_merged.count_female - df_merged.earnings_female
    df_merged['earn_hi_female'] = df_merged.earnings_female
    df_merged = df_merged.drop(columns=['count_male', 'count_female', 'earnings_male', 'earnings_female'])

    # Return the aggregated dataframe and 
    return df_merged, educationID_map, sexID_map

df, educationID_map, sexID_map = load_data()


# *************************************************************************************************
# 1.2. Following Example 10.1.3, build two models for the classification of an individual's yearly income 
# (1 being above $50k and 0 being below), 
# one of these models should include the effect of gender while the other should not.
