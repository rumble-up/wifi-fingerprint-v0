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
# %% Assumptions  --------------------------------------------------------------------

rand = 42


# Setup --------------------------------------------------------------------

# Change working directory to the folder where script is stored.
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

import pandas as pd
import xgboost as xgb
from pprint import pprint
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.externals import joblib

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

sample = '5_1_train_val'  # or 'all_train'

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


#  X for test/train for Floor, Latitude, and Longitude models ---------------
    
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
# ============================================================================
# %% Floor Random Forest Model --------------------------------------------

# Set the target variables for FLOOR
y_train, y_test, y_test2, y_train_final = set_y('FLOOR')

rf = RandomForestClassifier()

# PRINT MODEL DEFAULTS!
pprint(rf.get_params())

n_iter_search = 9
random_grid = {"n_estimators": [80, 100, 120],
              "max_features": ['auto', 'sqrt'],
              "max_depth": [10, 50, 100, None],
              "min_samples_leaf": [1, 2, 4],
              "min_samples_split":[2, 5, 10],
              "bootstrap": [True, False]}
#              "criterion": ["gini", "entropy"]}

rf_rscv = RandomizedSearchCV(rf, param_distributions=random_grid,
                             n_iter=n_iter_search, 
                             cv=5,
                             n_jobs = 2,
                             scoring = 'accuracy')

print("Performing randomized search for floor...")
print("Number of iterations:", n_iter_search)
search_time_start = time.time()
rf_rscv1 = rf_rscv.fit(X_train, y_train)
print("Randomized search time:", time.time() - search_time_start)


# %% Report Floor Model -------------------------------------------------------

print('**********************************************************************')
print('Floor model complete. \n')
acc_report(rf_rscv1, 'First model', True, 
           X_test, y_test, X_test2, y_test2)

# Choose the best estimator from Random Search as final model
rf_rscv_final = rf_rscv1.best_estimator_

# Take best model, fit with all available data
rf_rscv_final = rf_rscv_final.fit(X_train_final, y_train_final)


print('**********************************************************************')
print('Final model trained on full dataset')
acc_report(rf_rscv_final, 'Model trained on full set', False,
           X_test, y_test, X_test2, y_test2)

# Save model
model_name = 'rf_rscv_' + sample

joblib.dump(rf_rscv1, 'models/' + model_name + '_RSCV.sav')
joblib.dump(rf_rscv_final, 'models/' + model_name + '_final.sav')


# %% Final Floor prediction ---------------------------------------------------

# Be very careful changing this!!!
df_pred['FLOOR'] = rf_rscv_final.predict(X_pred_final)
df_pred  = df_pred.rename(columns = {'FLOOR': 'FLOOR_' + model_name + '_final.sav'})



# %% Lat/Long Regression Function --------------------------------------------

def lat_long_reg(target, tag, model_stop_pair, 
                 random_search, num_rounds, n_jobs, xgb_verbose,
                 df_pred, save_model
                 ):

    reg = xgb.XGBRegressor()


    random_grid = {'objective': ['reg:linear'],
               'n_estimators': [num_rounds],       #n_estimators is equivalent to num_rounds in alternative syntax
               'max_depth': [3, 7, 13, 16],
               'learning_rate': [0.05, 0.1, 0.15, 0.2]}
    
#    This was the old method, but it's deprecated 
#    fit_params = {'eval_metric': 'mae',
#              'early_stopping_rounds': 10,
#              'eval_set': [(X_test2, y_test2),]}

    xgb_rscv = RandomizedSearchCV(reg, random_grid,
                              n_iter=random_search,
                              n_jobs=n_jobs,
                              verbose=0,
                              cv=5,
#                              fit=fit_params,
                              scoring='neg_mean_absolute_error',
                              random_state=rand)

    print("Performing", target + '_' + tag, "randomized search...")
    print("Number of iterations:", random_search)
    search_time_start = time.time()
    xgb_rscv = xgb_rscv.fit(X_train, y_train,
                            # Last pair from eval_set is used to stop model early
                            eval_set = [(X_train, y_train), model_stop_pair,],
                            eval_metric='mae',
                            early_stopping_rounds=10,
                            verbose=xgb_verbose)
    
    print("Randomized search time:", (time.time() - search_time_start)/60, 'min')


    print(target + '_' + tag, "randomized search results ************************")
    mae_report(xgb_rscv, True, X_test, y_test, X_test2, y_test2)
    print('**********************************************************************')
    
    # Used for plotting learning curve
#    rscv_best_result = xgb_rscv.best_estimator_.evals_result()
    
    # Final LATITUDE/LONGITUDE model --------------------------------------------------

    # Take best model, fit with all available data
    xgb_rscv_final = xgb_rscv.best_estimator_
    
    print("Performing", target + '_' + tag, "final fit...")
    print("Number of iterations:", random_search)
    
    final_fit_time_start = time.time()
    xgb_rscv_final= xgb_rscv_final.fit(X_train_final, y_train_final,
            eval_set=[(X_train_final, y_train_final), model_stop_pair],
            eval_metric='mae',
            early_stopping_rounds=10,
            verbose=xgb_verbose)
    
    # Used for plotting learning curve
#    final_result = xgb_rscv_final.evals_result()
    
    print("Final fit time:", (time.time() - final_fit_time_start)/60, 'min')
    print('Final', target + '_' + tag, 'model results *************************************')
    mae_report(xgb_rscv_final, False, X_test, y_test, X_test2, y_test2)

    # Save LATITUDE/LONGITUDE model ------------------------------------------------------
                 # lat or lon  
    model_name = target[0:3].lower() +'_'+ tag + '_xgb_rscv_' + sample 
    
    if save_model:
        joblib.dump(xgb_rscv, 'models/' + model_name + '.sav')
        joblib.dump(xgb_rscv_final, 'models/' + model_name + '_final.sav')


    # Final LATITUDE/LONGITUDE prediction ---------------------------------------------------

    # Be very careful changing this!!!
    df_pred[target] = xgb_rscv_final.predict(X_pred_final)
    df_pred  = df_pred.rename(columns = {target: target + '_' + model_name + '_final.sav'})
    
    # Save csv before other predictions are ready
#    df_pred.to_csv('predictions/mvp_autotune_' + model_name +'.csv')
    
    return(df_pred)
#    return(rscv_best_result, final_result)

# %% Set parameters for both LAT/LONG predictions -----------------------------
tag = 'rand8'

random_search = 8
num_rounds = 500
# Number of cores to use on computer
n_jobs = 2 
xgb_verbose = False
save_model = True

# LATITUDE Predictions -----------------------------------------------------------
target = 'LATITUDE'

# Set the target variables to target
y_train, y_test, y_test2, y_train_final = set_y(target)

df_pred = lat_long_reg(target=target, tag=tag, 
             # Lack of improvement in this pair stops model training
             model_stop_pair=(X_test2, y_test2),
             random_search=random_search, num_rounds=num_rounds,
             n_jobs=n_jobs, 
             xgb_verbose=xgb_verbose, df_pred=df_pred,
             save_model=save_model)


# %% LONGITUDE Predictions -----------------------------------------------------------
target = 'LONGITUDE'

# Set the target variables to target
y_train, y_test, y_test2, y_train_final = set_y(target)

df_pred = lat_long_reg(target=target, tag=tag, 
             # Lack of improvement in this pair stops model training
             model_stop_pair=(X_test2, y_test2),
             random_search=random_search, num_rounds=num_rounds,
             n_jobs=n_jobs, 
             xgb_verbose=xgb_verbose, df_pred=df_pred,
             save_model=save_model)

# Export all predictions to csv
df_pred.to_csv('predictions/mvp_autotune_' + tag + '.csv')




# %% Visualize errors ---------------------------------------------------------

#error_both = errorLon + errorLat
#
## Tough test set
#y_test2_long = test_val['LONGITUDE']
#y_test2_lat = test_val['LATITUDE']
#
## Full test set
#y_test_long = test['LONGITUDE']
#y_test_lat = test['LATITUDE']
#
#def find_error(X_test, y_test, xgb_model):
#    dtest = xgb.DMatrix(X_test)
#    pred = xgb_model.predict(dtest)
#    error = pred - y_test
#    return(error)
#
## Error on tough test set only
#error_lat2 = find_error(X_test2, y_test2_lat, bst)
#error_lon2 = find_error(X_test2, y_test2_lat, bst_lon)
#
#error_lat = find_error(X_test, y_test_lat, bst)
#error_lon = find_error(X_test, y_test_long, bst_lon)
#
#
## Error on full test set
#error = error_lon
#y_plot = y_test_lat
#x_plot = y_test_long
#
## Error on tough test set
#error = error_lat2
#y_plot = y_test2_lat
#x_plot = y_test2_long
#
## Ensure that zero is always the same color, gray
#zero = abs(0 - min(error) / (max(error) - min(error)))
#
#colorscale = [[0, 'rgba(5,113,176, 1)'], 
#               [zero, 'rgba(211, 211, 211, 1)' ],
#               [1, 'rgba(202,0,32, 1)']]
#
#trace = go.Scatter3d(
#        x=x_plot,
#        y=y_plot,
#        z=error,
#        mode='markers',
#        marker = dict(
#                size = 4,
#                color=error,
#                colorscale=colorscale
#        )
#)
#
#plot([trace])



# %% BACKBURNER
# Plot learning curve ------------------------------------------------------

## This part not functional yet.
#result_rscv_best = rscv_best_result
#result_final = final_result
#
#result_final.values()
#
#
#
#def gen_traces(name, result_dict):
#    
#    all_traces = list()
#    
#    for validation in result_dict.keys():
#        
#        trace = go.Scatter(
#                name = name + '_' + validation,
#                y = result_dict[validation]['mae'])
#        all_traces.append(trace)
#    return(all_traces)
#
#        
#
#traces1 = gen_traces('rscv', result_rscv_best)
#traces2 = gen_traces('final', result_final)
#
#traces1.append(traces2.items())
#
#layout = go.Layout(
#        yaxis=dict(range=[0,20]))
#
#fig = go.Figure(data=traces1, layout=layout)
#plot(fig)
#
#
#
#results = result_final        
#
#trace1 = go.Scatter(
#        name = 'Train',
#        y = results['validation_0']['mae'])
#
#all_traces.append(trace1)
#
#type(trace1)
#
#data = list()
#data.append(trace1)
#
#
#trace2 = go.Scatter(
#        name = 'Test',
#        y = results['validation_1']['mae'])
#
#data = [trace1, trace2]
#
#layout = go.Layout(
#        yaxis=dict(range=[0,20]))
#
#fig = go.Figure(data=data, layout=layout)
#plot(fig)