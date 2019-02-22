#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: Use saved models to make final prediction
Status: IN PROGRESS

Created on Fri Feb 22 12:55:39 2019
@author: Laura Stupin

"""
# %% Setup --------------------------------------------------------------------

# Change working directory to the folder where script is stored.
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.externals import joblib

# %% Load processed data ------------------------------------------------------
df_all = pd.read_csv('data/processed/df.csv')
df_test = df_all[df_all.dataset == 'test']

# %% Floor model --------------------------------------------------------------

floor = joblib.load('models/rfc80final.sav')



floor.predict
