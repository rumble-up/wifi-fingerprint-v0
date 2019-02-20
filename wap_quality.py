#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Status: IN PROGRESS
Purpose: Which WAPs have the greatest range?

Created on Wed Feb 20 11:27:15 2019
@author: Laura Stupin

"""
# %% Model Settings  -------------------------------------------------

rand = 42          # Set the random seed number

# %% Model and Data Assumptions -------------------------------------------------

x100 = -110                 # Replace 100s with this value
drop_null_waps = True       # Drop WAPs not included in test set
drop_na_rows = True         # If no WAPs recorded, drop row
drop_duplicate_rows = False # 

# 76 na rows in train, but none in validation or test
#

# %% Setup --------------------------------------------------------------------

# Change working directory to the folder where script is stored.
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

import pandas as pd
import numpy as np
import pickle

import plotly.graph_objs as go
from plotly.offline import plot

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

import xgboost as xgb

#%% Load data -----------------------------------------------------------------

# Test data
df_train = pd.read_csv('data/trainingData.csv')
df_train['dataset'] = 'train'
# Validation data
df_val = pd.read_csv('data/validationData.csv')
df_val['dataset'] = 'validate'

df_test = pd.read_csv('data/testData.csv')
df_test['dataset'] = 'test'

df = pd.concat([df_train, df_val]).reset_index()

df = df.replace(100, np.nan)

wap_names = [col for col in df if col.startswith('WAP')]

lats = dict()
longs = dict()
wap_range = pd.DataFrame(index = wap_names,
                         columns = ['minLat', 'maxLat', 'minLon', 'maxLon'])

for w in wap_names:

    lats = df['LATITUDE'][df[w].notnull()]
    longs = df['LONGITUDE'][df[w].notnull()]
    wap_range.loc[w, 'minLat'] = min(lats)
    wap_range.loc[w, 'minLon'] = min(longs)
    wap_range.loc[w, 'maxLat'] = max(lats)
    wap_range.loc[w, 'maxLon'] = max(longs)
    
wap_range['lat_diff'] = wap_range.maxLat - wap_range.minLat
wap_range['long_diff'] = wap_range.maxLon - wap_range.minLon

trace1 = go.Histogram(x=wap_range.long_diff)
trace2 = go.Histogram(x=wap_range.lat_diff)
plot([trace1, trace2])
