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

top_num = 5        # Consider the top ___ Wifi signals
bot_num = 3        # Use the __ weakest Wifi signals
keep = ['LATITUDE', 'LONGITUDE', 'FLOOR', 'BUILDINGID', 'sig_count']


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

#%% Load data

df_raw = pd.read_csv('data/trainingData.csv')
wap_names = [col for col in df_raw if col.startswith('WAP')]

# Temporary - choose if working with full dataframe or subset
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

# Set seed
np.random.seed(42)

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


# %% Build dataframe with selected attributes ---------------------------------

# Change column names to be unique, remove last two characters
hi_rank.columns = [('hi' + str(name))[:-2] for name in hi_rank.columns]
low_rank.columns = [('lo' + str(name))[:-2] for name in low_rank.columns]

# Select only the top __ and the bottom ___
ranks = hi_rank.ix[:, 0:top_num].join(low_rank.ix[:, 0:bot_num])
# Change to categorical variables
ranks = ranks.apply(lambda x: x.astype('category'))

# Compile full data frame
df_rank = df[keep].join(ranks)
df_rank.dtypes


#%% Sandbox/Archive


