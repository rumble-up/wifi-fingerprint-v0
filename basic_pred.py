#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Status: IN PROGRESS
Purpose: Use raw signals to make model prediction

Created on Tue Feb 19 17:42:43 2019
@author: Laura Stupin

"""
# %% Model Settings  -------------------------------------------------

rand = 42          # Set the random seed number

# %% Model and Data Assumptions -------------------------------------------------

x100 = -110                 # Replace 100s with this value
drop_null_waps = True       # Drop WAPs not included in both train and validation
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


#%% Load data -----------------------------------------------------------------

# Test data
df_train = pd.read_csv('data/trainingData.csv')
df_train['dataset'] = 'train'
# Validation data
df_val = pd.read_csv('data/validationData.csv')
df_val['dataset'] = 'validation'

df_test = pd.read_csv('data/testData.csv')
df_test['dataset'] = 'test'

# Combine datasets so identical pre-processing will happen to all
df = pd.concat([df_train, df_val, df_test]).reset_index()

# Load previous lists
with open('data/wap_groups.pkl', 'rb') as f:
    wap_groups = pickle.load(f)
    
null_waps = wap_groups['null_train'] + wap_groups['null_test']

# Drop null WAPs
if drop_null_waps: df = df.drop(null_waps, axis=1)

# Collect valid WAP names
wap_names = [col for col in df if col.startswith('WAP')]

# Switch values
df = df.replace(100, np.nan)
df['sig_count'] = len(wap_names) - df[wap_names].isnull().sum(axis=1)

# There are 76 NA rows, they appear only in training set
if drop_na_rows: df = df[df['sig_count'] != 0]
if drop_duplicate_rows: df = df.drop_duplicates()

# Replace Na's with the selected number
df = df.replace(np.nan, x100)