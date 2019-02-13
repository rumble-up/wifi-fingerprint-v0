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
# %% Assumptions in these calculations --------------------------------------------------------------
unknown_na = True # All 100 values are replaced by NaNs

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

if unknown_na: df = df.replace(100, np.nan)

# Count observations per row
df['sig_count'] = 520 - df[wap_names].isnull().sum(axis=1)

# %% Rank signals with random tie-breaking ------------------------------------

# Set seed
np.random.seed(42)

# Create small random noise 
noise = np.random.rand(*df[wap_names].shape) / 10000.0 

# Add noise to each signal so that ties are broken randomly
# Rank along each row
rank = (df[wap_names] + noise).rank(axis=1)

# Melt into long form, order by row number and rank
rank = pd.melt(rank.reset_index(), id_vars='index').sort_values(by=['index', 'value'])

# Drop na values
rank = rank[rank['value'].notna()]

# Pivot back to have the columns be first rank, second rank, etc
rank = rank.pivot(index = 'index', columns = 'value', values = 'variable')




trace = go.Histogram(x=df['sig_count'])
plot([trace])

#%% Sandbox/Archive

# Example of ranking within row with nans
foo = pd.DataFrame(dict(WAP001 = [2, 5, 17],
                   WAP002 = [3, np.nan, 1],
                   WAP003 = [15, np.nan, np.nan]))

rank = foo.rank(axis=1)
rank2 = pd.melt(rank.reset_index(), id_vars='index').sort_values(by=['index', 'value'])
rank2 = rank2[rank2['value'].notna()]
rank3 = rank2.pivot(index = 'index', columns = 'value', values = 'variable')
rank3


foo = foo.transpose()
foo['rank'] = foo.sort_values(by=)
