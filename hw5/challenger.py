"""
Michael S. Emanuel
Tue Oct  9 22:14:15 2018
"""

# *************************************************************************************************
# Question 2: The Consequences of O-ring Failure can be Painful and Deadly
# *************************************************************************************************

# In 1986, the space shuttle Challenger exploded during take off, killing the seven astronauts aboard. 
# It is believed that the explosion was caused by the failure of an O-ring 
# (a rubber ring that seals parts of the solid fuel rockets together), 
# and that the failure was caused by the cold weather at the time of launch (31F).

# In the file chall.txt, you will find temperature (in Fahrenheit) and failure data from 23 shuttle launches, 
# where 1 stands for O-ring failure and 0 no failure. We assume that the observed temperatures are fixed and that,  
# at temperature tt, an O-ring fails with probability f(θ1+θ2t)conditionally on Θ=(θ1,θ2)

# f(z) is defined to be the logistic function -- f(z)=1/(1+exp(−z))

# 2.1. Based on your own knowledge and experience, suggest a prior distribution for the regression parameters (Θ1,Θ2). 
# Make sure to explain your choice of prior.

# 2.2. Produce 5000-10000 samples from the posterior distribution of Θ using rejection sampling, 
# and plot them and their marginals. (This may take a while.)

# 2.3. Use the logit package in the statsmodels library to compute 68% confidence intervals on the θ parameters. 
# Compare those intervals with the 68% credible intervals from the posterior above. 
# Overlay these on the above marginals plots.

# 2.4. Use the MLE values from statsmodels and the posterior mean from 2.2 at each temperature to plot the probability
#  of failure in the frequentist and bayesian settings as a function of temperature. What do you see?

# 2.5. Compute the mean posterior probability for an O-ring failure at t=31∘Ft=31∘F. 
# To do this you must calculate the posterior at 31∘F31∘F and take the mean of the samples obtained.

# 2.6. You can instead obtain the probability from the posterior predictive. 
# Use the posterior samples to obtain samples from the posterior predictive at 31∘F 
# and calculate the fraction of failures.

# 2.7. The day before a new launch, meteorologists predict that the temperature will be T∼N(68,1) during take-off. 
# Estimate the probability for an O-ring failure during this take-off. 
# (You will calculate multiple predictives at different temperatures for this purpose).