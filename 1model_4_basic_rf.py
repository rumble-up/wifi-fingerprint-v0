#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Status: IN PROGRESS
Purpose: Predict Wifi location in test competition set
    + Import pre-processed data from 0data.py, 
    + Autotune models

Future steps: 
    + CV evaluate on full data set before predicting unknown test
    + Clean up global/local variables
    
Created on Feb 22 2019
@author: Laura Stupin

"""
# %% Parameters for running models --------------------------------------------------------------------

rand = 42
n_jobs = 3
lon_lat_rand_search = 8
floor_rand_search = 6


# Setup --------------------------------------------------------------------

# Change working directory to the folder where script is stored.
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

import pandas as pd

from pprint import pprint
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.externals import joblib

from scipy.spatial import distance
import plotly.graph_objs as go
from plotly.offline import plot

# Custom functions ----------------------------------------------------------------

# Calculate and display error metrics for Random Forest Classification
def rfc_pred(X_test, y_test, model):
    clfpred = model.predict(X_test)
    print(pd.crosstab(y_test, clfpred, rownames=['Actual'], colnames=['Predicted']))
    print('Accuracy:', model.score(X_test, y_test))
    print('Kappa:', cohen_kappa_score(clfpred, y_test))

# Set up y for each target - Floor, Latitude, Longitude
def set_y(target):
    y_train = train[target]   
    y_test = test[target]

    # A more difficult test
    y_test2 = test_val[target]
    y_train_final = train_final[target]
    return(y_train, y_test, y_test2, y_train_final)
    
# Define an accuracy report for Random Forest classification models
def acc_report(model, tag, is_search, X_test, y_test, X_test2, y_test2):
    accuracy = dict(test_acc = model.score(X_test, y_test),
                    test_kappa = cohen_kappa_score(model.predict(X_test), y_test),
                    test2_acc = model.score(X_test2, y_test2),
                    test2_kappa = cohen_kappa_score(model.predict(X_test2), y_test2))
    if is_search:
        print(tag, 'BEST PARAMETERS')
        pprint(model.best_params_)
    print('\n', tag, 'FULL TEST')
    print(pd.crosstab(y_test, model.predict(X_test), rownames=['Actual'], colnames=['Predicted']))
    print('\n', tag, ' TEST WITH VALIDATION ONLY')
    print(pd.crosstab(y_test2, model.predict(X_test2), rownames=['Actual'], colnames=['Predicted']))
    pprint(accuracy)

def mae_report(model, is_search, X_test, y_test, X_test2, y_test2):
    
    if is_search: 
        best_score = model.best_score_
        best_params = model.best_params_
        print("Cross Validation Scores:", model.cv_results_['mean_test_score'])
        print("Best score: {}".format(best_score))
        print("Best params: ")
        for param_name in sorted(best_params.keys()):
            print('%s: %r' % (param_name, best_params[param_name]))
    
    mae = dict(test_mae = mean_absolute_error(model.predict(X_test), y_test),
               test2_mae = mean_absolute_error(model.predict(X_test2), y_test2)
               )
             
    pprint(mae)
# Load data -------------------------------------------------------

df_all = pd.read_csv('data/processed/df.csv')
df_tr = df_all[df_all['dataset'] == 'train']
df_val = df_all[df_all['dataset'] == 'val']
df_test = df_all[df_all['dataset'] == 'test']

wap_names = [col for col in df_all if col.startswith('WAP')]

# Empty dataframe to hold final predictions
df_pred = pd.DataFrame(
        index = range(0,len(df_test)),
        columns = ['FLOOR', 'LATITUDE', 'LONGITUDE'])

# Prepare test/train -------------------------------------------------------

sample = 'all_train'  # '5_1_train_val' or 'all_train'

# less_train has 400*13 = 5200 train samples
#                1110 all validation, so 5:1 weight train to validation
if sample == '5_1_train_val':

    # Build a random sample of val data, 25 from each floor/building
    test_val = df_val.groupby(['BUILDINGID', 'FLOOR']).apply(lambda x: x.sample(n=20, random_state=rand))
    test_val = test_val.droplevel(level = ['BUILDINGID', 'FLOOR'])
    # Old way, random sampling all data
    # test_val = df_val.sample(n = 250, random_state = rand)
    # The rest is training
    train_val = df_val.drop(test_val.index)
    
    # Build a random sample with 400 observations from each floor, building 
    tr_samp = df_tr.groupby(['BUILDINGID', 'FLOOR']).apply(lambda x: x.sample(n = 400, random_state = rand))
    # Reduce multi-index to single level index
    tr_samp = tr_samp.droplevel(level = ['BUILDINGID', 'FLOOR'])
    test_tr = tr_samp.sample(n=round(.25*len(tr_samp)), random_state = rand)
    
    train_tr = tr_samp.drop(test_tr.index)
    
    # Build the final test/train sets from both
    test = pd.concat([test_val, test_tr])
    train = pd.concat([train_val, train_tr])
    train_final = pd.concat([df_val, tr_samp])

if sample == 'all_train':
    # Build a random sample of val data alone
    test_val = df_val.sample(n = 250, random_state = rand)
    
    # Build a random test sample with 400 observations from each floor, building 
    test = df_tr.groupby(['BUILDINGID', 'FLOOR']).apply(lambda x: x.sample(n = 400, random_state = rand))
    # Reduce multi-index to single level index
    test = test.droplevel(level = ['BUILDINGID', 'FLOOR'])
    
    # Put both random samples into the main test set
    test = pd.concat([test, test_val])
    # Training is all observations not in random test sample or provided test set
    train = df_all[df_all['dataset'] != 'test'].drop(test.index)
    
    # Use all available data for final prediction
    train_final = pd.concat([df_tr, df_val])


# %% X for test/train for Floor, Latitude, and Longitude models ---------------
    
# Set up x for all predictions
X_train = train[wap_names]
X_test = test[wap_names]
# A more difficult test
X_test2 = test_val[wap_names]

# Use all available data to train for final predictions
X_train_final = train_final[wap_names]

## The WAPS needed to make the final prediction
X_pred_final = df_test[wap_names]
  
#df_full = df_all[df_all['dataset'] != 'test']
# ****************************************************************************
# =============================================================================
# %% Building/Floor Random Forest Classifier function --------------------------------------------


def rf_bld_flr(target, rand_search, n_jobs, save_model):
    
    
    # Set the target variables for FLOOR
    y_train, y_test, y_test2, y_train_final = set_y(target)

    rf = RandomForestClassifier()

    random_grid = {"n_estimators": [80, 100, 120],
                  "max_features": ['auto', 'sqrt'],
                  "max_depth": [10, 50, 100, None],
                  "min_samples_leaf": [1, 2, 4],
                  "min_samples_split":[2, 5, 10],
                  "bootstrap": [True, False]}
    #              "criterion": ["gini", "entropy"]}

    rf_rscv = RandomizedSearchCV(rf, param_distributions=random_grid,
                             n_iter=rand_search, 
                             cv=5,
                             n_jobs = n_jobs,
                             scoring = 'accuracy')

    print("Performing randomized search for", target + "...")
    print("Number of iterations:", rand_search)
    search_time_start = time.time()
    rf_rscv = rf_rscv.fit(X_train, y_train)
    print("Randomized search time:", time.time() - search_time_start)


    # Report Floor Model -------------------------------------------------------
    
    print('**********************************************************************')
    print(target, 'model complete. \n')
    acc_report(rf_rscv, 'First model', True, 
               X_test, y_test, X_test2, y_test2)

    # Choose the best estimator from Random Search as final model
    rf_rscv_final = rf_rscv.best_estimator_
    
    # Take best model, fit with all available data
    rf_rscv_final = rf_rscv_final.fit(X_train_final, y_train_final)


    print('**********************************************************************')
    print('Final model trained on full dataset')
    acc_report(rf_rscv_final, 'Model trained on full set', False,
               X_test, y_test, X_test2, y_test2)

    # Save model
    model_name = target + '_rand'+ str(rand_search) + '_rf_rscv_' + sample
    
    if save_model:
        joblib.dump(rf_rscv, 'models/' + model_name + '.sav')
        joblib.dump(rf_rscv_final, 'models/' + model_name + '_final.sav')


    # Final Floor prediction ---------------------------------------------------

    # Be very careful changing this!!!
    df_pred[target] = rf_rscv_final.predict(X_pred_final)
#    df_pred  = df_pred.rename(columns = {'FLOOR': 'FLOOR_' + model_name + '_final.sav'})

    return(rf_rscv, rf_rscv_final)
# %% Building prediction ---------------------------------------------------------

# 100% accuracy when rand_search=3
bld_model, bld_model_final = rf_bld_flr(target='BUILDINGID', 
                                        rand_search=3, 
                                        n_jobs=n_jobs, 
                                        save_model=True)
    
# %% Add BUILDINGID to predictors ---------------------------------------------

def add_predictor(X_df, new_col_name, model):
    X_df[new_col_name] = model.predict(X_df)
    return(X_df)

new_col_name = 'bld_pred'

X_train = add_predictor(X_train, new_col_name, bld_model)   
X_test = add_predictor(X_test, new_col_name, bld_model)
X_test2 = add_predictor(X_test2, new_col_name, bld_model)
X_train_final = add_predictor(X_train_final, new_col_name, bld_model)

# Use the final model to predict final building
X_pred_final = add_predictor(X_pred_final, new_col_name, bld_model_final)
 

# %% Floor prediction ---------------------------------------------------------
flr_model, flr_model_final = rf_bld_flr(target='FLOOR', 
                                    rand_search=floor_rand_search, 
                                    n_jobs=n_jobs, 
                                    save_model=True)

# %% Add FLOOR to predictors

new_col_name = 'flr_pred'
model = flr_model

X_train = add_predictor(X_train, new_col_name, model)   
X_test = add_predictor(X_test, new_col_name, model)
X_test2 = add_predictor(X_test2, new_col_name, model)
X_train_final = add_predictor(X_train_final, new_col_name, model)

# Use the final model to predict final building
X_pred_final = add_predictor(X_pred_final, new_col_name, 
                             flr_model_final)

# %% Longitude/Latitude Random Forest Regressor Function -----------------------

def rf_lon_lat(target, rand_search, n_jobs, save_model):
    
    
    # Set the target variables for FLOOR
    y_train, y_test, y_test2, y_train_final = set_y(target)

    rf = RandomForestRegressor()

    random_grid = {"n_estimators": [80, 100, 120],
                  "max_features": ['auto', 'sqrt'],
                  "max_depth": [10, 50, 100, None],
                  "min_samples_leaf": [1, 2, 4],
                  "min_samples_split":[2, 5, 10],
                  "bootstrap": [True, False]}
    #              "criterion": ["gini", "entropy"]}

    rf_rscv = RandomizedSearchCV(rf, param_distributions=random_grid,
                             n_iter=rand_search, 
                             cv=5,
                             n_jobs = n_jobs,
                             scoring = 'explained_variance')

    print("Performing randomized search for", target + "...")
    print("Number of iterations:", rand_search)
    search_time_start = time.time()
    rf_rscv = rf_rscv.fit(X_train, y_train)
    print("Randomized search time minutes:", (time.time() - search_time_start)/60)
 
    print('**********************************************************************')
    print(target, 'model complete. \n')
    
    # Choose the best estimator from Random Search as final model
    rf_rscv_final = rf_rscv.best_estimator_
    
    # Take best model, fit with all available data
    rf_rscv_final = rf_rscv_final.fit(X_train_final, y_train_final)

    print('**********************************************************************')
    print('Final model trained on full dataset')

    # Save model
    model_name = target + '_rand'+ str(rand_search) + '_rf_rscv_' + sample
    
    if save_model:
        joblib.dump(rf_rscv, 'models/' + model_name + '.sav')
        joblib.dump(rf_rscv_final, 'models/' + model_name + '_final.sav')

    # Final TARGET prediction ---------------------------------------------------

    # Be very careful changing this!!!
    df_pred[target] = rf_rscv_final.predict(X_pred_final)
#    df_pred  = df_pred.rename(columns = {'FLOOR': 'FLOOR_' + model_name + '_final.sav'})

    return(rf_rscv, rf_rscv_final, model_name)

# %% Predict LONGITUDE --------------------------------------------------------

lon_lat_rand_search = 9

lon_model, lon_model_final, model_name = rf_lon_lat(target='LONGITUDE', 
                                    rand_search=lon_lat_rand_search, 
                                    n_jobs=n_jobs, 
                                    save_model=True)


# %% Predict LATITUDE --------------------------------------------------------
lat_model, lat_model_final, model_name = rf_lon_lat(target='LATITUDE', 
                                    rand_search=lon_lat_rand_search, 
                                    n_jobs=n_jobs, 
                                    save_model=True)

# %% Export all predictions to csv --------------------------------------------

# Export all predictions to csv
df_pred.to_csv('predictions/'+ model_name +'.csv', index=False)


# %% Old Experiments Below ----------------------------------------------------
# =============================================================================

# %% Remove floor as a predictor

col_name = 'flr_pred'

X_train = X_train.drop(col_name, axis=1)  
X_test = X_test.drop(col_name, axis=1) 
X_test2 = X_test2.drop(col_name, axis=1) 
X_train_final = X_train_final.drop(col_name, axis=1) 

X_pred_final = X_pred_final.drop(col_name, axis=1) 

# %% Predict LONGITUDE --------------------------------------------------------



lon_model, lon_model_final = rf_lon_lat(target='LONGITUDE', 
                                    rand_search=lon_lat_rand_search, 
                                    n_jobs=3, 
                                    save_model=True)


# Predict LATITUDE --------------------------------------------------------
lat_model, lat_model_final = rf_lon_lat(target='LATITUDE', 
                                    rand_search=lon_lat_rand_search, 
                                    n_jobs=3, 
                                    save_model=True)

# Export all predictions to csv --------------------------------------------

# Export all predictions to csv
df_pred.to_csv('predictions/RF_no_flr_pred_rand' + str(lon_lat_rand_search) + '.csv', index=False)

# %% Visualize errors ---------------------------------------------------------

# Tough test set
y_test2_long = test_val['LONGITUDE']
y_test2_lat = test_val['LATITUDE']

# Full test set
y_test_long = test['LONGITUDE']
y_test_lat = test['LATITUDE']

def find_error(X_test, y_test, model):
    pred = model.predict(X_test)
    error = pred - y_test
    return(error)
    
# Error on tough test set only
error_lat2 = find_error(X_test2, y_test2_lat, lat_model)
error_lon2 = find_error(X_test2, y_test2_lat, lon_model)

error_lat = find_error(X_test, y_test_lat, lat_model)
error_lon = find_error(X_test, y_test_long, lon_model)


# Error on full test set
error = error_lon
y_plot = y_test_lat
x_plot = y_test_long

# Error on tough test set
error = error_lat2
y_plot = y_test2_lat
x_plot = y_test2_long

# Ensure that zero is always the same color, gray
zero = abs(0 - min(error) / (max(error) - min(error)))

colorscale = [[0, 'rgba(5,113,176, 1)'], 
               [zero, 'rgba(211, 211, 211, 1)' ],
               [1, 'rgba(202,0,32, 1)']]

tr_AE = go.Scatter3d(
        x=x_plot,
        y=y_plot,
        z=abs(error),
        mode='markers',
        marker = dict(
                size = 4,
                color=error,
                colorscale=colorscale
        )
)
        
layout = go.Layout(dict(
        title="Test Set Longitude Error",
        titlefont=dict( size=40)))

fig = go.Figure(data = [tr_AE],
                layout = layout)

plot(fig)
