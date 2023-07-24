# -*- coding: utf-8 -*-
"""
##################################################
#
# ECO 5445: Business Analytics
#
# Joshua Eubanks
# Instructor
# Department of Economics
# College of Business 
# University of Central Florida
#
# July 11, 2022
# 
##################################################
#
# Basic statistics and visualization
#
##################################################
"""

##################################################
# Bringing in modules for this script
##################################################

import os # for setting up proper working directory
import pandas as pd # for reading and viewing data
import matplotlib.pyplot as plt # for visualizing data
import seaborn as sns # another way to view data
import numpy as np # to build arrays
import scipy as sp # for descriptive statistics
import statsmodels.stats.weightstats as sm # more descriptive statistics

##################################################
# Set Working Directory.
##################################################


# Find out the current directory.
os.getcwd()
# Change to a new directory.
git_path = 'C:\\Users\\jo585802\\OneDrive - University of Central Florida\\Documents\\GitHub\\ECO5445\\'
os.chdir(git_path + '\\12-BasicStatsAndVisualization\\data')
# Check that the change was successful.
os.getcwd()

##################################################
# Bring in data
##################################################

housing = pd.read_csv("prop_prices.csv")

##################################################
#
# Summary statistics
#
##################################################

##### Via numpy #####

x_bar = np.mean(housing["sale_def"])
x_median = np.median(housing.sale_def) # showing another way to call columns
s_x_bar = np.std(housing["sale_def"])
s_squared = np.var(housing["sale_def"])

# Could clean up these values
round(x_bar, -4)
round(s_x_bar, -4)

# Checking correlation
r = np.corrcoef(housing["sale_def"],housing["area_heated"])

# you can also calculate correlation between multiple variables
variables = [housing["sale_def"],housing["area_heated"],housing["bed"], housing["bath"]]

r = np.corrcoef(variables)

##### Via Pandas #####

# There are some basic statistics items contained within a pandas dataframe

summary = housing.describe()

# Perhaps you want variance of each of the columns

var = summary.iloc[2,]**2
var

# You can calculate the summary statistic for one or more columns

housing["sale_def"].mean() 
housing[["sale_def","bed"]].median() # note the extra brackets due to inserting list 

# You can also group by other variables

# Perhaps you want to look at the median price difference between homes with
# a pool and ones without

housing[["sale_def", "pool"]].groupby("pool").median()

# or show the distribution of homes with pools by the number of beds
housing[["bed", "pool"]].groupby("bed").count()

# Calculate correlation
housing.corr()

##### Via Scipy #####
sp.stats.describe(housing["sale_def"])

sp.stats.pearsonr(housing["sale_def"], housing["area_heated"])
# (0.692010800241011, 2.0608780707706464e-143) output is (correlation coef, p-value)

##### Via Statsmodels #####
summary_sale_def = sm.DescrStatsW(housing["sale_def"])

# This creates an object to which you can pull from
summary_sale_def.std
summary_sale_def.mean

# Can perform hypothesis tests
h_0 = 200000
summary_sale_def.ttest_mean(h_0)
# (0.03498638855175941, 0.9720975837471996, 999.0) # output is (t-stat, p-value, df)

# Manually calculating
t_stat = (summary_sale_def.mean - h_0)/(summary_sale_def.std/(np.sqrt(summary_sale_def.nobs - 1)))
p_value = sp.stats.t.sf(abs(t_stat),(summary_sale_def.nobs - 1))*2 # Since two tailed
df = summary_sale_def.nobs - 1

##################################################
#
# Visualizing Data
#
##################################################

##### Via matplotlib #####

x = np.random.randn(1000) # Create an array of normally distributed values

plt.plot(x)
plt.hist(x) # Very wide bins
plt.hist(x, bins = 50)
plt.hist(x, bins = "fd") # Some strings also work https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
plt.hist(x, bins = "fd",cumulative=True)

plt.scatter(housing["area_heated"], housing["sale_def"])
plt.hist(housing["sale_def"], bins = "fd")
plt.hist(housing["sale_def"], bins = "fd",cumulative=True)

##### Via Pandas #####

# There are many different ploting options on your pandas dataframe


"""
['area',
 'bar',
 'barh',
 'box',
 'density',
 'hexbin',
 'hist',
 'kde',
 'line',
 'pie',
 'scatter']
"""
housing.plot()
housing.plot(subplots=True)

housing_subset = housing[["bed","bath"]]
housing_subset.plot.box(figsize=(1.68*5,5),subplots = True) # 1.68 is for golden ratio: https://en.wikipedia.org/wiki/Golden_ratio

##### Via Seaborn #####

sns.relplot(x="area_heated", y="sale_def", hue="pool", size = "area_heated", data=housing,alpha = 0.5);

sns.displot(x="area", hue = "pool",data = housing)
sns.displot(housing["sale_def"])

sns.displot(housing["bath"], discrete=True)

housing_quantitative_subset = housing[["sale_def","bed","bath","area_heated","area"]]
sns.pairplot(housing_quantitative_subset, corner = True)

