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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.externals import joblib

# %% Custom functions ----------------------------------------------------------------

# Calculate and display error metrics for Random Forest Classification
def rfcPred(X_test, y_test, model):
    clfpred = model.predict(X_test)
    print(pd.crosstab(y_test, clfpred, rownames=['Actual'], colnames=['Predicted']))
    print('Accuracy:', model.score(X_test, y_test))
    print('Kappa:', cohen_kappa_score(clfpred, y_test))

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



# %% Floor Random Forest Model ------------------------------------------------
# Y TEST/TRAIN DATA -----------------------------------------------------------
target = 'FLOOR'

y_train = train[target]   
y_test = test[target]

# A more difficult test
y_test2 = test_val[target]
y_train_final = train_final[target]

# %% Floor Random Forest Model --------------------------------------------
# Model training and prediction

rfc80 = RandomForestClassifier(n_estimators = 80, n_jobs =2, random_state=rand)
rfc80 = rfc80.fit(X_train, y_train)

# Look at error metrics
rfcPred(X_test, y_test, rfc80)
# Try the harder test set
rfcPred(X_test2, y_test2, rfc80)

# Train model on full dataset and save for final prediction
rfc80final = RandomForestClassifier(n_estimators = 80, n_jobs =2, random_state=rand)
rfc80final = rfc80final.fit(X_train_final, y_train_final)

#Save model for reference
joblib.dump(rfc80, 'models/rfc80train.sav')
joblib.dump(rfc80final, 'models/rfc80final.sav')

# Be very careful changing this!!!
df_pred['FLOOR'] = rfc80final.predict(X_pred_final)
df_pred.to_csv('predictions/marshmellow.csv')

# %% Latitude XGB Model --------------------------------------------
# TEST/TRAIN DATA #### ------------------------------------------------------------

target = 'LATITUDE'

y_train = train[target]   
y_test = test[target]

# A more difficult test
y_test2 = test_val[target]


y_train_final = df_full[target]

# %% Latitude XGB Model --------------------------------------------
# Train/fit model with normal test data ------------------------------------------------------------

# Define function to try out different test/train splits
def xgb_fit(X_train, y_train, X_test, y_test, param):
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
         'n_estimators':100,
         'early_stopping_rounds':10}
    
# Tougher test set
bst = xgb_fit(X_train, y_train, X_test2, y_test2, param)    

bst = xgb_fit(X_train, y_train, X_test, y_test, param)

# Output array of predictions


bst.save_model('models/lat_xgb_best_tough.model')

errorLat = xgbpredLat - y_test

trace1 = go.Scatter(
        x=y_test,
        y=errorLat,
        mode='markers'
)


print('The Latitude MAE is:', abs(xgbpredLat - y_test).mean())

# %% Latitude XGB Model --------------------------------------------
# Tough test data: Train/fit model ------------------------------------------------------------

# 400 was plenty
num_round = 400
param = {'objective':'reg:linear',
         'max_depth':10, 
         'learning_rate': 0.3,
         'n_estimators':50,
         'early_stopping_rounds':10}
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
evallist = [(dtest, 'eval'), (dtrain, 'train')]


bst = xgb.train(param, dtrain, num_round, evallist)


# Output array of predictions
xgbpredLat = bst.predict(dtest)
mean_absolute_error(xgbpredLat, y_test)

bst.save_model('models/lat_xgb.model')

errorLat = xgbpredLat - y_test

trace1 = go.Scatter(
        x=y_test,
        y=errorLat,
        mode='markers'
)


print('The Latitude MAE is:', abs(xgbpredLat - y_test).mean())


# %% Longitude XGB model ------------------------------------------------------
X = df_full[wap_names]
y = df_full['LONGITUDE']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.25, random_state = rand)


param = {'max_depth':6, 'objective':'reg:linear'}


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

evallist = [(dtest, 'eval'), (dtrain, 'train')]

# 400 was plenty
num_round = 600
bst = xgb.train(param, dtrain, num_round, evallist)
bst.save_model('models/xgbLong500_drop_noiWAP.model')


# Output array of predictions
xgbpredLong = bst.predict(dtest)


errorLong = xgbpredLong - y_test

trace2 = go.Scatter(
        x=y_test,
        y=errorLong,
        mode='markers')

plot([trace2])



print('The Longitude MAE is:', abs(xgbpred - y_test).mean())
