#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Status: IN PROGRESS
Purpose: Reformat data by rank from UJIIndoorLoc dataset

Created on Tue Feb 12 16:56:40 2019
@author: Laura Stupin

Known issues:
  * Rank method - if two signals equal -84 and -84 is it fair to give them separate ranks? 

"""
# %% Model Settings  -------------------------------------------------

top_num = 10       # Consider the top ___ Wifi signals
bot_num = 1        # Use the __ weakest Wifi signals
keep = ['LATITUDE', 'LONGITUDE', 'FLOOR', 'BUILDINGID', 'sig_count']
s_num = 0        # Samples per building/floor, if 0, use all data

# NOT FIXED FOR SAMPLING YET
rand = 42          # Set the random seed number
 # Set seed for random tie-breaking


# %% Model and Data Assumptions -------------------------------------------------

x100_to_na = True                 # All 100 values are replaced by NaNs

drop_null_training_waps = False   # Shouldn't matter for rank method

drop_na_rows = True  
'''DECISION: Drop.  Assume NaNs are a function of phone, not location.
FREQUENCY: 76 of 19,937 rows full of NaNs
Mostly from PHONEID 1, seems like other datapoints for those locations available  
But is no signal a function of the location or the phone?
For now, assume it's produced by the phone, drop NaN rows
'''
  
drop_duplicate_rows = False
'''DECISION: Keep duplicates.
FREQUENCY: 637 of 19,861 rows contain duplicated information.
It looks like for some locations, 20/40 observations are duplicates
This indicates to me that a single phone took 10 exactly same measurements
Assumption - it's better to keep 20 measurements per point, 
even if half of them are identical.
COUNTER-ARGUMENT:  Should one measurement be weighted 10 times for a location?
This actually may not be a problem for the rank method

EXAMPLE: single location with 10/20 duplicated observations  
bar = df[(df['LONGITUDE'] == -7390.761199999601) & 
          (df['LATITUDE'] == 4864835.141000004) &
          (df.SPACEID == 147)]
barred = bar[bar.duplicated()]
'''
# %% Setup --------------------------------------------------------------------

# Change working directory to the folder where script is stored.
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

import pandas as pd
import numpy as np

import plotly.graph_objs as go
from plotly.offline import plot

from sklearn.ensemble import RandomForestClassifier

#%% Load data

df_raw = pd.read_csv('data/trainingData.csv')
wap_names = [col for col in df_raw if col.startswith('WAP')]

# Select subset or sample of full data, if any
if s_num > 0:
    # Sample by building/floor
    df = df_raw.groupby(['BUILDINGID', 'FLOOR']).apply(lambda x: x.sample(s_num))
    df = df.reset_index(drop=True)
else:
    df = df_raw
    




if x100_to_na: df = df.replace(100, np.nan)


# Count observations per row
df['sig_count'] = 520 - df[wap_names].isnull().sum(axis=1)
trace = go.Histogram(x=df['sig_count'])
plot([trace])

# Implement assumptions
if drop_na_rows: df = df[df['sig_count'] != 0]
if drop_duplicate_rows: df = df.drop_duplicates()



# %% Rank signals with random tie-breaking ------------------------------------

np.random.seed(rand)

# Add noise to each signal so that ties are broken randomly 
noisy = df[wap_names] + (np.random.rand(*df[wap_names].shape) / 10000.0) 

# ---------- Rank Strongest Signals ----------
# Rank along each row
hi_rank = noisy.rank(axis=1, ascending=False)

# Melt into long form, order by row number and rank (easier to see order and debug)
hi_rank = pd.melt(hi_rank.reset_index(), id_vars='index').sort_values(by=['index', 'value'])

# Drop na values
hi_rank = hi_rank.dropna()

# Pivot back to have the columns be first rank, second rank, etc
hi_rank = hi_rank.pivot(index = 'index', columns = 'value', values = 'variable')

# ---------- Rank Weakest Signals ----------

low_rank = noisy.rank(axis=1, ascending=True)
low_rank = pd.melt(low_rank.reset_index(), id_vars='index').dropna()
low_rank = low_rank.pivot(index = 'index', columns = 'value', values = 'variable')

# Change column names to be unique, remove last two characters
hi_rank.columns = [('hi' + str(name))[:-2] for name in hi_rank.columns]
low_rank.columns = [('lo' + str(name))[:-2] for name in low_rank.columns]

# %% Build dataframe with selected attributes ---------------------------------


# Select only the top __ and the bottom ___
ranks = hi_rank.iloc[:, 0:top_num].join(low_rank.iloc[:, 0:bot_num])
# Change to categorical variables
ranks = ranks.apply(lambda x: x.astype('category'))

# Compile full data frame
df_rank = df[keep].join(ranks)
df_rank[['BUILDINGID', 'FLOOR']] = df_rank[['BUILDINGID', 'FLOOR']].apply(lambda x: x.astype('category'))
df_rank.dtypes

# %% Build Decision Tree Model ------------------------------------------------

# Roughly 227 categories

# Albon try

y =pd.factorize(df_rank['FLOOR'])[0]

# Create a list of factors we're looking at
factors = [('hi' + str(i)) for i in range(1, top_num+1)]


df_big = pd.DataFrame()

for f in factors:
    dft = pd.get_dummies(df_rank[f])
    dft.columns = [(f + str(col)) for col in dft.columns]
    df_big = df_big.join(dft, how = 'outer')


# Can only use categorical now.

# Create random forest classifier
# n_jobs = 2 -> use two processors
clf = RandomForestClassifier(n_jobs=2, random_state = rand)
clf.fit(df_big, y)
train_rez = clf.predict(df_big)
pd.crosstab(y, train_rez, rownames=['Actual'], colnames=['Predicted'])

clf.score(df_big,y)




list(zip(both, clf.feature_importances_))



# Categorical variables
# http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder
# https://stackoverflow.com/questions/38108832/passing-categorical-data-to-sklearn-decision-tree

# Random forest example
# http://localhost:8888/notebooks/Dropbox/0Python/courses/1701-portilla/Lec%2086%20-%20Decision%20Trees%20and%20Random%20Forests.ipynb



# %% Sandbox/Archive

# Storing mutliple variables!!!
# https://stackoverflow.com/questions/2960864/how-can-i-save-all-the-variables-in-the-current-python-session

from sklearn.datasets import load_iris

iris = load_iris()
iris.target
