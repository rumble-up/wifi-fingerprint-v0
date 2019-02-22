#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Status: In Progress
Purpose: Simple random forest function

Created on Feb 22 2019

@author: Laura Stupin
"""

# %% Assumptions  --------------------------------------------------------------------

rand = 42

# %% Setup --------------------------------------------------------------------

# Change working directory to the folder where script is stored.
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from time import time



# %% Prepare test/train

df_all = pd.read_csv('data/processed/df.csv')

wap_names = [col for col in df_all if col.startswith('WAP')]

df_tr = df_all[df_all['dataset'] == 'train']

df_val = df_all[df_all['dataset'] == 'val']


# Build a random sample of val data alone
test2 = df_val.sample(n = 250, random_state = rand)
   

# Build a random test sample with 400 observations from each floor, building 
test = df_tr.groupby(['BUILDINGID', 'FLOOR']).apply(lambda x: x.sample(n = 400, random_state = rand))
# Reduce multi-index to single level index
test = test.droplevel(level = ['BUILDINGID', 'FLOOR'])
# Put both random samples into the main test set
test = pd.concat([test, test2])

# Training is all observations not in random test sample or provided test set
train = df_all[df_all['dataset'] != 'test'].drop(test.index)


### CHANGE TO FUNCTION #########################################################
target = 'LATITUDE'

y_train = train[target]   
X_train = train[wap_names]

y_test = test[target]
X_test = test[wap_names]



# Build classifier
reg = RandomForestRegressor(max_depth=6, n_estimators=30)
# specify parameters and distributions to sample from
#param_dist = {"max_depth": [3, 5],
#              "bootstrap": [True, False],
#              "criterion": ["mse", "mae"],
#              "n_estimators": [20, 50, 100]}

# Grid search

rfLat1 = reg.fit(X_train, y_train)
rf_pred_Lat = rfLat1.predict(X_test)
mean_absolute_error(rf_pred_Lat, y_test)

rf_train_lat = rfLat1.predict(X_train)
mean_absolute_error(rf_train_lat, y_train)

# %% Random grid search -------------------------------------------------------

n_iter = 5
rf = RandomForestRegressor()

random_grid = {"max_depth": [6, 10],
#              "bootstrap": [True, False],
#              "criterion": ["mse", "mae"],
              "n_estimators": [20, 50, 100]}

rf_rsCV = RandomizedSearchCV(estimator = rf, 
                             param_distributions = random_grid,
                             n_iter=n_iter,
                             cv=5,
                             random_state = rand,
                             n_jobs = 1,
                             scoring = 'neg_mean_absolute_error')
rf_rsCV.fit(X_train, y_train)

lat_pred = rf_rsCV.predict(X_test)
mean_absolute_error(lat_pred, y_test)

lat_train = rf_rsCV.predict(X_train)
mean_absolute_error(lat_train, y_train)

