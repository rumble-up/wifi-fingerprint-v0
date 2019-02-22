#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Status: In Progress
Purpose: Grid search different machine learning models

Created on Wed Feb 20 16:44:05 2019

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
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split


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

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from time import time
from scipy.stats import randint as sp_randint
import numpy as np



# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# Build classifier
reg = RandomForestRegressor()
# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, 5],
              "bootstrap": [True, False],
              "criterion": ["mse", "mae"],
              "n_estimators": [20, 50, 100]}

# Grid search
grid_search = GridSearchCV(estimator=reg,
                           param_grid=param_dist,
                           cv=5,
                           scoring='neg_mean_absolute_error',
                           n_jobs=2)

gsCVlat1 = grid_search.fit(X_train, y_train)

# run randomized search
n_iter_search = 2
random_search = RandomizedSearchCV(estimator=clf, 
                                   param_distributions=param_dist,
                                   n_iter=n_iter_search, 
                                   cv=5,
                                   scoring='neg_mean_absolute_error')

start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

random_search.score(X_train, y_train)
random_search.cv_results_
random_search.scorer_

# This was taking suspiciously long to execute
# https://jessesw.com/XG-Boost/

cv_params = {'max_depth': [3,5]} #,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 600, 'seed': rand, 
          'subsample': 0.8, 'colsample_bytree': 0.8, 
         'objective': 'reg:linear'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), cv_params, 
                         scoring = 'neg_mean_absolute_error', 
                         cv = 5, n_jobs = 2) 
    

lat1 = optimized_GBM.fit(X_train, y_train)
