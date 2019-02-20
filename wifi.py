#%%
# Computational power

#%%
## Importing Packages
import numpy as np  # for calculations
import pandas as pd  # for dataframes
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
#from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelBinarizer
#from sklearn.model_selection import learning_curve
#from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt
plt.close('all')
from mpl_toolkits.mplot3d import Axes3D 
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn

import os
import csv 


import pickle
#%%
## Get and change working directory
os.getcwd()
os.chdir('/Users/denizminican/Dropbox/03-Data_and_Coding/Ubiqum/Repositories/wifi-fingerprint')

#%%
## Create the df
train = pd.read_csv('data/trainingData.csv')
# Load validation set
val = pd.read_csv('data/validationData.csv')

#%%
### PREPROCESSING
## Changing 100s to something else
train = train.replace(100, -110)
val = val.replace(100, -110)

## Removing zero variance columns
train2 = train.loc[:, train.apply(pd.Series.nunique) != 1]
val2 = val.loc[:, val.apply(pd.Series.nunique) != 1]

# Finding common APs and getting them to a list
tra_cols = list(train2.columns.values)
val_cols = list(val2.columns.values)
cols312 = list(item for item in val_cols if item in tra_cols)
# Removing extra APs from the sample_train to have 312 APs
train2 = train2[cols312] 
val2 = val2[cols312] 

# Sampling for train
sample_train = train2.groupby(['BUILDINGID','FLOOR']).apply(lambda x: x.sample(n=900)) #.reset_index(drop=True)
sample_train.index.names

sample_train = sample_train.droplevel(level= ['FLOOR', 'BUILDINGID']) # clear the additional indexes

# Creating training and test sets
training = sample_train.append(val2)
validating = train2.drop(sample_train.index)
validating = validating.sample(frac=0.3, replace=True, random_state=1) # t was too large

# Prepare train and val waps objects
train_waps = training.iloc[:, :312]
val_waps = validating.iloc[:, :312]
val_build = validating.loc[:, 'BUILDINGID']
val_floor = validating.loc[:,'FLOOR']
val_lat = validating.loc[:,'LATITUDE']
val_long = validating.loc[:,'LONGITUDE']

# Reorder columns
train_dep = training.iloc[:, 312:]
cols = train_dep.columns.tolist()
cols = [cols[3]] + [cols [2]] + cols [0:2] + cols[4:]
train_dep = train_dep[cols]
training = pd.concat([train_waps, train_dep], axis=1)

# Removing values stronger than -30


#%%
### PREPROCESSING - TRAINING SET 2 - Taking top k signal WAPs from each row 

## Some loop to choose the top APs
# =============================================================================
# # K i want for the job
# k = 10  
# # Preparing the dfs to fill
# topdf_values = pd.DataFrame(columns=range(k),index=range(19937))
# 
# # Loop for taking the signal strengths
# for i in range(len(train.index)):
#     topdf_values.iloc[i, :] = train.iloc[i, :520].sort_values(axis = 0, ascending = False)[:k]
# 
# # Loop for taking the WAP numbers
# topdf_ranks = train.iloc[:, :520].apply(lambda s: s.nlargest(10).index.tolist(), axis=1)
# topdf_ranks_df = topdf_ranks.to_frame()
# 
# deneme = pd.DataFrame(train.iloc[0, :520].sort_values(axis = 0, ascending = False)[:k])
# 
# =============================================================================
# Cbind row and row2

#%%
### VISUALIZATION
## Initial plot
list(train)
mainplot = train.plot.scatter(x= "LATITUDE", y= "LONGITUDE")

## Finding the entries of users seperately
users = np.unique(train["USERID"])
plots = {}
for user in users: 
    subset = train.loc[train["USERID"] == user,:]
    plots[user] = subset.plot.scatter(x= "LATITUDE", y= "LONGITUDE")

## Plot with color per user
train.plot.scatter(x= "LATITUDE", y= "LONGITUDE", c= "USERID")  # how to put colors instead of grayscale

## PLot 3D with floors
tdplot = plt.figure().gca(projection='3d')
tdplot.scatter(xs=train["LATITUDE"], ys=train["LONGITUDE"], zs=train["FLOOR"], c=train["FLOOR"])
tdplot.set_zlabel('Floor')
plt.show()
plt.figure()

#%%
### DATA EXPLORATION
# PhoneID dBm boxplots
train_phone = train.drop(['LATITUDE', 'LONGITUDE', 'USERID', 'FLOOR', 'BUILDINGID', 'SPACEID',
                          'RELATIVEPOSITION', 'TIMESTAMP'], axis = 1)
melted_phone = pd.melt(train_phone, id_vars=['PHONEID'], var_name='WAP')
melted_phone = melted_phone[melted_phone['value'] != 100]
melted_phone.boxplot(by = "PHONEID", column = "value")

#%%
### DATA EXPLORATION
## WAP distribution per building and floor
# Melt for long format
melted_table = pd.melt(train, id_vars=['PHONEID', 'LATITUDE', 'LONGITUDE', 'USERID', 
                                       'FLOOR', 'BUILDINGID', 'SPACEID', 'RELATIVEPOSITION', 
                                       'TIMESTAMP'], var_name='WAP')
# Remove unused WAPs
melted_table = melted_table[melted_table['value'] != 100]

# Check buildings and floors
melted_table['FLOOR'].unique()
buildings = melted_table['BUILDINGID'].unique()

# Melted buildings
# =============================================================================
# buildings = [0,1,2]
# building_list = []
# for i in buildings:
#     building_list[i] = melted_table[melted_table['BUILDINGID'] == i]
# =============================================================================
    
building0 = melted_table[melted_table['BUILDINGID'] == 0]
building1 = melted_table[melted_table['BUILDINGID'] == 1]
building2 = melted_table[melted_table['BUILDINGID'] == 2] 
wap0 = building0['WAP'].unique().tolist()
wap1 = building1['WAP'].unique().tolist()
wap2 = building2['WAP'].unique().tolist()

# Same APs that are showing up in different buildings
wap01 = list(set(wap0) & set(wap1))  
wap12 = list(set(wap1) & set(wap2))
wap02 = list(set(wap0) & set(wap2))

#%%
# Saving objects to file
# =============================================================================
# if not wap_building:
#     wap_building = dict(wap01 = wap01, wap12 = wap12, wap02 = wap02, wap0 = wap0, wap1 = wap1, wap2 = wap2)
# 
# if 
# with open('data/wap_buildings.pkl', 'wb') as f:
#     pickle.dump(wap_building, f)
# =============================================================================

#%%
### VISUALIZATION
# Frequency of the WAPs
wapcount = melted_table['WAP'].value_counts()
wapcount.plot.hist()

py.iplot(go.Histogram())  # ValueError: The first argument to the 
# plotly.graph_objs.Histogram constructor must be a dict or an instance of plotly.graph_objs.Histogram 

#%%
### VISUALIZATION 
# Melt for long format
melted_val = pd.melt(val, id_vars=['PHONEID', 'LATITUDE', 'LONGITUDE', 
                                       'FLOOR', 'BUILDINGID', 
                                       'TIMESTAMP'], var_name='WAP')
# Remove unused WAPs
melted_val = melted_val[melted_val['value'] != 100]

tra_wap = melted_table['WAP'].unique().tolist()  # 465, correct
val_wap = melted_val['WAP'].unique().tolist()  # 370, 3 additional 

# Finding the common WAPs
common_waps = list(set(tra_wap) & set(val_wap))  # 312, correct

# Finding the common location points
tra_loc = melted_table.drop_duplicates(subset= ['LATITUDE', 'LONGITUDE'])  # ask if correct
tra_loc = tra_loc.drop(['SPACEID', 'RELATIVEPOSITION', 'USERID'], axis = 1)
val_loc = melted_val.drop_duplicates(subset= ['LATITUDE', 'LONGITUDE'])

#%%
### VISUALIZATION
# Joining them for plotting in same plot
tra_loc['dataset'] = 'training'
val_loc['dataset'] = 'validation'
all_loc = pd.concat([tra_loc, val_loc])

tra_loc.plot.scatter(x= "LATITUDE", y= "LONGITUDE")
val_loc.plot.scatter(x= "LATITUDE", y= "LONGITUDE")

# Training and validation unique points plot
fg = seaborn.FacetGrid(data=all_loc, hue='dataset')
fg.map(plt.scatter, 'LATITUDE', 'LONGITUDE').add_legend()



#%%
## Convert building and floor to categoric ()
# =============================================================================
# train['BUILDINGID'] = train['BUILDINGID'].astype('category')
# train['FLOOR'] = train['FLOOR'].astype('category')
# =============================================================================

#%%
#### MODELING

### 1- BUILDING - Models on Preprocessed Data

# Remove the target variables from the training set
y1 = sample_train.pop('BUILDINGID').values
y2 = sample_train.pop('FLOOR').values
y3 = sample_train.pop('LATITUDE').values
y4 = sample_train.pop('LONGITUDE').values

## Features and targets normalization (scaling)

## Cross validation

# Creating the objects of the classifiers
svc1 = LinearSVC(random_state=0, max_iter= 10000, verbose=2)  # verbose=2 to see the progress
knn1 = KNeighborsClassifier(n_neighbors=3)
xgb1 = XGBClassifier()
rf1 = RandomForestClassifier()

## SVM
# Train on training data and predict using the testing data
fit_svc1 = svc1.fit(train_waps, y1)
pred_svc1 = fit_svc1.predict(val_waps)
svc1.score(train_waps, y1)
accuracy_score(val_build, pred_svc1)

## KNN
# Train on training data and predict using the testing data
fit_knn1 = knn1.fit(train_waps, y1)
pred_knn1 = fit_knn1.predict(val_waps)
knn1.score(train_waps, y1)
accuracy_score(val_build, pred_knn1)

## XGBoost
# Train on training data and predict using the testing data
fit_xgb1 = xgb1.fit(train_waps, y1)
pred_xgb1 = fit_xgb1.predict(val_waps)
xgb1.score(train_waps, y1)
accuracy_score(val_build, pred_xgb1)

## RandomForest
# Train on training data and predict using the testing data
fit_rf1 = rf1.fit(train_waps, y1)
pred_rf1 = fit_rf1.predict(val_waps)
rf1.score(train_waps, y1)
accuracy_score(val_build, pred_rf1)

# Confusion matrix
pd.crosstab(val_build, pred_rf1)
#%%
### 2- FLOOR - Models on Preprocessed Data

## XGBoost
# Train on training data and predict using the testing data
fit_xgb2 = xgb1.fit(train_waps, y2)
pred_xgb2 = fit_xgb2.predict(val_waps)
xgb1.score(train_waps, y2)
accuracy_score(val_floor, pred_xgb2)

## RandomForest
# Train on training data and predict using the testing data
fit_rf2 = rf1.fit(train_waps, y2)
pred_rf2 = fit_rf2.predict(val_waps)
rf1.score(train_waps, y2)
accuracy_score(val_floor, pred_rf2)

# Confusion matrix
pd.crosstab(val_floor, pred_rf2)
#%%
### 3- LATITUDE - Models on Preprocessed Data

## XGBoost
# Train on training data and predict using the testing data
fit_xgb3 = xgb1.fit(train_waps, y3)
pred_xgb3 = fit_xgb3.predict(val_waps)
xgb1.score(train_waps, y3)
accuracy_score(val_lat, pred_xgb3)

## RandomForest
# Train on training data and predict using the testing data
fit_rf3 = rf1.fit(train_waps, y3)
pred_rf3 = fit_rf3.predict(val_waps)
rf1.score(train_waps, y3)
accuracy_score(val_lat, pred_rf3)

#%%
### 3- LONGITUDE - Models on Preprocessed Data

## XGBoost
# Train on training data and predict using the testing data
fit_xgb4 = xgb1.fit(train_waps, y4)
pred_xgb4 = fit_xgb4.predict(val_waps)
xgb1.score(train_waps, y4)
accuracy_score(val_long, pred_xgb4)

## RandomForest
# Train on training data and predict using the testing data
fit_rf4 = rf1.fit(train_waps, y4)
pred_rf4 = fit_rf4.predict(val_waps)
rf1.score(train_waps, y4)
accuracy_score(val_long, pred_rf4)

#%%
# Save models to file
# models = dict(fit_svc1 = fit_svc1, fit_knn1 = fit_knn1, fit_xgb1)
# 
# with open('data/wap_buildings.pkl', 'wb') as f:
#     pickle.dump(wap_building, f)

