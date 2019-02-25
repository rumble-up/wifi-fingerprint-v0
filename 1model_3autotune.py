#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Status: IN PROGRESS
Purpose: Use raw signals to make model prediction

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
from pprint import pprint
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.externals import joblib

import plotly.graph_objs as go
from plotly.offline import plot

# %% Custom functions ----------------------------------------------------------------

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

rf_rscv1 = rf_rscv.fit(X_train, y_train)

# Define an accuracy report for models
def acc_report(model, tag, X_test, y_test, X_test2, y_test2):
    accuracy = dict(test_acc = model.score(X_test, y_test),
                    test_kappa = cohen_kappa_score(model.predict(X_test), y_test),
                    test2_acc = model.score(X_test2, y_test2),
                    test2_kappa = cohen_kappa_score(model.predict(X_test2), y_test2))
    
    print(tag, 'BEST PARAMETERS')
    pprint(model.best_params_)
    print('\n', tag, 'FULL TEST')
    print(pd.crosstab(y_test, model.predict(X_test), rownames=['Actual'], colnames=['Predicted']))
    print('\n', tag, ' TEST WITH VALIDATION ONLY')
    print(pd.crosstab(y_test2, model.predict(X_test2), rownames=['Actual'], colnames=['Predicted']))
    pprint(accuracy)


print('**********************************************************************')
print('Floor model complete. \n')
acc_report(rf_rscv1, 'First model', X_test, y_test, X_test2, y_test2)

model_name = 'rf_rscv'

joblib.dump(rf_rscv1, 'models/' + model_name + '_train.sav')


print('**********************************************************************')
print('\n Training final model... ')

rf_rscv_final = rf_rscv.fit(X_train_final, y_train_final)
print('Floor final model complete. \n')
acc_report(rf_rscv_final, 'FINAL MODEL', X_test, y_test, X_test2, y_test2)

joblib.dump(rf_rscv_final, 'models/' + model_name + '_final.sav')


# Be very careful changing this!!!
df_pred['FLOOR'] = rf_rscv_final.predict(X_pred_final)
df_pred  = df_pred.rename(columns = {'FLOOR': 'FLOOR_' + model_name + 'final.sav'})

# Save csv before other predictions are ready
df_pred.to_csv('predictions/mvp_autotune_overfit.csv')


# %% Latitude XGB Model --------------------------------------------

# https://gist.github.com/wrwr/3f6b66bf4ee01bf48be965f60d14454d

# Set the target variables for LATITUDE
y_train, y_test, y_test2, y_train_final = set_y('LATITUDE')

reg = xgb.XGBRegressor()

        
n_iter_search = 6
random_grid = {'objective': ['reg:linear'],
               'silent': [False],
               'n_estimators': [350],       #n_estimators is equivalent to num_rounds in alternative syntax
               'max_depth': [3, 7, 13, 16],
               'learning_rate': [0.1, 0.24, 0.3]}

fit_params = {'eval_metric': 'mae',
              'early_stopping_rounds': 10,
              'eval_set': [(X_test2, y_test2)]}

reg_rscv = RandomizedSearchCV(reg, random_grid,
                              n_iter=n_iter_search,
                              n_jobs=2,
                              verbose=2,
                              cv=5,
                              fit_params=fit_params,
                              scoring='neg_mean_absolute_error',
                              random_state=rand)

print("LATITUDE randomized search...")
search_time_start = time.time()
reg_rscv = reg_rscv.fit(X_train, y_train)
print("Randomized search time:", time.time() - search_time_start)

best_score = reg_rscv.best_score_
best_params = reg_rscv.best_params_
pprint(reg_rscv.cv_results_['mean_test_score'])
print("Best score: {}".format(best_score))
print("Best params: ")
for param_name in sorted(best_params.keys()):
    print('%s: %r' % (param_name, best_params[param_name]))


f0 = reg_rscv.predict(X_test2)

mean_absolute_error(f0, y_test2)
# %% Leftovers from old Latitude

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
df_pred.to_csv('predictions/marshmellow_latitude.csv')

dtest2 = xgb.DMatrix(X_test2)
mvp1_lat = bst.predict(dtest2)
errorLat = mvp1_lat - y_test2


trace1 = go.Scatter(
        x=y_test2,
        y=errorLat,
        mode='markers'
)

plot([trace1])
#
#
#print('The Latitude MAE is:', abs(xgbpredLat - y_test).mean())


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

# Calculate errors/residuals on toughest test set
dtest2 = xgb.DMatrix(X_test2)
mvp1_lon = bst_lon.predict(dtest2)
errorLon = mvp1_lon - y_test2

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




#
#errorLong = xgbpredLong - y_test
#
#trace2 = go.Scatter(
#        x=y_test,
#        y=errorLong,
#        mode='markers')
#
#plot([trace2])
#
#
#
#print('The Longitude MAE is:', abs(xgbpred - y_test).mean())
