#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: MVP - Use raw signals to make model prediction
+ Uses a "smart" train/test set that will more likely mimic unknown data

Status: Script works, but not polished


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
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.externals import joblib

import plotly.graph_objs as go
from plotly.offline import plot

# %% Custom functions ----------------------------------------------------------------

# Calculate and display error metrics for Random Forest Classification
def rfcPred(X_test, y_test, model):
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

# %% Load data -------------------------------------------------------

df_all = pd.read_csv('data/processed/df.csv')
df_tr = df_all[df_all['dataset'] == 'train']
df_val = df_all[df_all['dataset'] == 'val']
df_test = df_all[df_all['dataset'] == 'test']

wap_names = [col for col in df_all if col.startswith('WAP')]

# Empty dataframe to hold final predictions
df_pred = pd.DataFrame(
        index = range(0,len(df_test)),
        columns = ['FLOOR', 'LATITUDE', 'LONGITUDE'])

# %% Prepare test/train -------------------------------------------------------

sample = 'less_train'  # or 'all_train'

if sample == 'less_train':

    # Build a random sample of val data alone
    test_val = df_val.sample(n = 250, random_state = rand)
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


# %% Floor Random Forest Model --------------------------------------------

# Set the target variables for FLOOR
y_train, y_test, y_test2, y_train_final = set_y('FLOOR')

rfc80 = RandomForestClassifier(n_estimators = 80, n_jobs =2, random_state=rand)
rfc80 = rfc80.fit(X_train, y_train)

# Look at error metrics
rfcPred(X_test, y_test, rfc80)
# Try the harder test set
rfcPred(X_test2, y_test2, rfc80)

# Train model on full dataset and save for final prediction
rfc80final = RandomForestClassifier(n_estimators = 80, n_jobs =2, random_state=rand)
rfc80final = rfc80final.fit(X_train_final, y_train_final)

model_name = 'floor_rfc80'

#Save model for reference
joblib.dump(rfc80, 'models/' + model_name + '_train.sav')
joblib.dump(rfc80final, 'models/' + model_name + '_final.sav')

# Be very careful changing this!!!
df_pred['FLOOR'] = rfc80final.predict(X_pred_final)
df_pred  = df_pred.rename(columns = {'FLOOR': 'FLOOR_' + model_name + 'final.sav'})

# %% Latitude XGB Model --------------------------------------------

# Set the target variables for LATITUDE
y_train, y_test, y_test2, y_train_final = set_y('LATITUDE')

# Define function to try out different test/train splits
def xgb_fit(X_train, y_train, X_test, y_test, param):
    
    # This is the required format for pure xgb without sklearn API
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    
    bst = xgb.train(param, dtrain, num_round, evallist)
    
    xgbpred = bst.predict(dtest)
    print('MAE:', mean_absolute_error(xgbpred, y_test))
      
    return(bst)

# 400 was plenty
num_round = 250
param = {'objective':'reg:linear',
         'max_depth':13, 
         'learning_rate': 0.24,
#         'n_estimators':100,
         'early_stopping_rounds':10}
    
# Tougher test set
bst = xgb_fit(X_train, y_train, X_test2, y_test2, param)    

bst = xgb_fit(X_train, y_train, X_test, y_test, param)



bst_final = xgb_fit(X_train_final, y_train_final, X_test2, y_test2, param)

model_name = 'lat_xgb_tough'

bst.save_model('models/'+ model_name + '_train.model')
bst_final.save_model('models/'+ model_name + '_final.model')

# Make final prediction
dtest_final = xgb.DMatrix(X_pred_final)
df_pred['LATITUDE'] = bst_final.predict(dtest_final)
df_pred  = df_pred.rename(columns = {'LATITUDE': 'LATITUDE_' + model_name + '_final.model'})


# %% Longitude XGB model ------------------------------------------------------

# Set the target variables for LATITUDE
y_train, y_test, y_test2, y_train_final = set_y('LONGITUDE')

num_round = 450
param = {'objective':'reg:linear',
         'max_depth':7, 
         'learning_rate': 0.3,
         'gamma':5,
         'early_stopping_rounds':10}

# Tougher test set
bst_lon = xgb_fit(X_train, y_train, X_test2, y_test2, param)
#Easier test set
bst_lon = xgb_fit(X_train, y_train, X_test, y_test, param)    


# Make final prediction
bst_final_lon = xgb_fit(X_train_final, y_train_final, X_test2, y_test2, param)

model_name = 'lon_xbg_tough'

bst_lon.save_model('models/'+ model_name + '_train.model')
bst_final.save_model('models/'+ model_name + '_final.model')

dtest_final = xgb.DMatrix(X_pred_final)
df_pred['LONGITUDE'] = bst_final_lon.predict(dtest_final)
df_pred  = df_pred.rename(columns = {'LONGITUDE': 'LONGITUDE_' + model_name + '_final.model'})


# %% Save predictions to CSV --------------------------------------------------
df_pred.to_csv('predictions/marshmellow_all2.csv')

# %% Visualize errors ---------------------------------------------------------

error_both = errorLon + errorLat

# Tough test set
y_test2_long = test_val['LONGITUDE']
y_test2_lat = test_val['LATITUDE']

# Full test set
y_test_long = test['LONGITUDE']
y_test_lat = test['LATITUDE']

def find_error(X_test, y_test, xgb_model):
    dtest = xgb.DMatrix(X_test)
    pred = xgb_model.predict(dtest)
    error = pred - y_test
    return(error)

# Error on tough test set only
error_lat2 = find_error(X_test2, y_test2_lat, bst)
error_lon2 = find_error(X_test2, y_test2_lat, bst_lon)

error_lat = find_error(X_test, y_test_lat, bst)
error_lon = find_error(X_test, y_test_long, bst_lon)


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

trace = go.Scatter3d(
        x=x_plot,
        y=y_plot,
        z=error,
        mode='markers',
        marker = dict(
                size = 4,
                color=error,
                colorscale=colorscale
        )
)

plot([trace])


