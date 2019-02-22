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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import mean_absolute_error
from sklearn.externals.joblib import dump, load
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
from pathlib import Path

import pickle
#%%
## Get and change working directory
os.getcwd()
os.chdir('/Users/denizminican/Dropbox/03-Data_and_Coding/Ubiqum/Repositories/wifi-fingerprint')

#%%
## Load training set
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
# Drop levels to be able to identify rows ands divide it later
sample_train.index.names
sample_train = sample_train.droplevel(level= ['FLOOR', 'BUILDINGID']) # clear the additional indexes

## Creating training and test sets
# Separating the validation set to two
val2_1 = val2.sample(frac=0.5, replace= True, random_state=32)
val2_2 = val2.drop(val2_1.index)  # why does it give 670 instead of 555???????????????
# Create train set with half validation
training = sample_train.append(val2_1)
# Create validation set with rest
validating = train2.drop(sample_train.index)
validating = validating.sample(frac=0.3, replace=True, random_state=33) # t was too large
validating = validating.append(val2_2)

train_waps = training.iloc[:, :312]
val_waps = validating.iloc[:, :312]

# Reorder columns
train_dep = training.iloc[:, 312:]
val_dep = validating.iloc[:, 312:]
cols = train_dep.columns.tolist()
cols = [cols[3]] + cols [0:3] + cols[4:]
train_dep = train_dep[cols]
val_dep = val_dep[cols]
training = pd.concat([train_waps, train_dep], axis=1)
validating = pd.concat([val_waps, val_dep], axis=1)

# Prepare train and val waps objects
train_wapsb = training.iloc[:, :313]
val_wapsb = validating.iloc[:, :313]
train_wapsblo = training.iloc[:, :314]
val_wapsblo = validating.iloc[:, :314]

# Objects to be used for prediction performances
val_build = validating.loc[:, 'BUILDINGID']
val_floor = validating.loc[:,'FLOOR']
val_lat = validating.loc[:,'LATITUDE']
val_long = validating.loc[:,'LONGITUDE']

train_lat = training.loc[:,'LATITUDE']
train_long = training.loc[:,'LONGITUDE']

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
melted_phone = melted_phone[melted_phone['value'] != -110]
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
#### MODELING ####

### 1- BUILDING - Models on Preprocessed Data

# Remove the target variables from the training set
y1 = training['BUILDINGID']
y2 = training['FLOOR']
y3 = training['LONGITUDE']
y4 = training['LATITUDE']
yy = pd.concat([y3, y4], axis=1)

## Features and targets normalization (scaling)

# Creating the objects of the classifiers
svc1 = LinearSVC(random_state=0, max_iter= 10000, loss= 'hinge', verbose=2)  # verbose=2 to see the progress
knn1 = KNeighborsClassifier(n_neighbors=3)
xgb1 = XGBClassifier(n_jobs=-1, verbose=2)
rf1 = RandomForestClassifier(max_features= 'sqrt' ,n_estimators=100, verbose=2)
rfr = RandomForestRegressor(n_estimators=200, n_jobs=2, verbose=2)

## Cross validation RFC
rfcv = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True, verbose=2) 
param_grid = { 
    'n_estimators': [100, 250],
    'max_features': ['auto', 'sqrt', 'log2']
}
CV_rfcv = GridSearchCV(estimator=rfcv, param_grid=param_grid, cv=5)

# =============================================================================
# ## Cross validation RFC
# rfcv = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True, verbose=2) 
# param_grid = { 
#     'n_estimators': [100, 250],
#     'max_features': ['auto', 'sqrt', 'log2']
# }
# CV_rfcv = GridSearchCV(estimator=rfcv, param_grid=param_grid, cv=5)
# =============================================================================

## SVM
# Train on training data and predict using the testing data
train_waps_scaled = scale(train_waps)
fit_svc1 = svc1.fit(train_waps_scaled, y1)
val_waps_scaled = scale(val_waps)
pred_svc1 = fit_svc1.predict(val_waps_scaled)
svc1.score(train_waps_scaled, y1)
accuracy_score(val_build, pred_svc1)

## KNN
# Train on training data and predict using the testing data
fit_knn1 = knn1.fit(train_waps, y1)
pred_knn1 = fit_knn1.predict(val_waps)
knn1.score(train_waps, y1)
accuracy_score(val_build, pred_knn1)
cohen_kappa_score(val_build, pred_knn1)
      
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
cohen_kappa_score(val_build, pred_rf1)

# Confusion matrix
pd.crosstab(val_build, pred_knn1)
#%%
#### MODELING ####

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
cohen_kappa_score(val_floor, pred_rf2)

# Confusion matrix
pd.crosstab(val_floor, pred_rf2)
#%%
#### MODELING ####

### 3- LONGITUDE - Models on Preprocessed Data
# Add building predictions to dataframe
val_wapsb_pred = val_waps.copy()
val_wapsb_pred.loc[:, "BUILDINGID"] = pred_knn1

## XGBoost
xgb3_file = Path("xgb3.joblib")
if xgb3_file.is_file():
    print ("XGB3 here.")
    fit_xgb3 = load('xgb3.joblib')
else:
    fit_xgb3 = xgb1.fit(train_wapsb, y3)
    dump(fit_xgb3, 'xgb3.joblib')
# Prediction
pred_xgb3 = fit_xgb3.predict(val_wapsb_pred)
mean_absolute_error(val_long, pred_xgb3)

## RandomForest
rf3_file = Path("rf3.joblib")
if rf3_file.is_file():
    print ("RF3 here.")
    fit_rf3 = load('rf3.joblib')  
else:
    fit_rf3 = rfr.fit(train_wapsb, y3)
    dump(fit_rf3, 'rf3.joblib') 
# Prediction
pred_rf3 = fit_rf3.predict(val_wapsb_pred)
mean_absolute_error(val_long, pred_rf3)

#%%
#### MODELING ####

### 4- LATITUDE - Models on Preprocessed Data
# Add longitude predictions to dataframe
val_wapsblo_pred = val_wapsb_pred.copy()
val_wapsblo_pred.loc[:, "LONGITUDE"] = pred_rf3.copy()

## XGBoost
xgb4_file = Path("xgb4.joblib")
if xgb4_file.is_file():
    print ("I'm here.")
    fit_xgb4 = load('xgb4.joblib')
else:
    fit_xgb4 = xgb1.fit(train_wapsblo, y4)
    dump(fit_xgb4, 'xgb4.joblib')
# Prediction
pred_xgb4 = fit_xgb4.predict(val_wapsblo_pred)
mean_absolute_error(val_lat, pred_xgb4)

## XGBoost
xgb4_2_file = Path("xgb4_2.joblib")
if xgb4_2_file.is_file():
    print ("XGB4-2 here.")
    fit_xgb4_2 = load('xgb4_2.joblib') 
else:
    fit_xgb4_2 = xgb1.fit(train_wapsb, y4)
    dump(fit_xgb4_2, 'xgb4_2.joblib') 
# Prediction
pred_xgb4_2 = fit_xgb4_2.predict(val_wapsb_pred)
mean_absolute_error(val_lat, pred_xgb4_2)

## RandomForest
rf4_file = Path("rf4.joblib")
if rf4_file.is_file():
    print ("RF4 here.")
    fit_rf4 = load('rf4.joblib') 
else:
    fit_rf4 = rfr.fit(train_wapsblo, y4)
    dump(fit_rf4, 'rf4.joblib')  
# Prediction
pred_rf4 = fit_rf4.predict(val_wapsblo_pred)
mean_absolute_error(val_lat, pred_rf4)

## RandomForest2
rf4_2_file = Path("rf4_2.joblib")
if rf4_2_file.is_file():
    print ("RF4-2 here.")
    fit_rf4_2 = load('rf4_2.joblib') 
else:
    fit_rf4_2 = rfr.fit(train_wapsb, y4)
    dump(fit_rf4_2, 'rf4_2.joblib') 
# Prediction
pred_rf4_2 = fit_rf4_2.predict(val_wapsb_pred)
mean_absolute_error(val_lat, pred_rf4_2)

#%%
#### TEST DATA STUFF ####

# Load test set
test = pd.read_csv('data/testData.csv')

## Finding common APs
aps_train = train.iloc[:, :520]
aps_val = val.iloc[:, :520]
aps_test = test.iloc[:, :520]
aps_train2 = train.loc[:, train.apply(pd.Series.nunique) != 1]
aps_val2 = val.loc[:, val.apply(pd.Series.nunique) != 1]
aps_test2 = test.loc[:, test.apply(pd.Series.nunique) != 1]

aps_tetr = np.intersect1d(aps_test.columns, aps_train.columns)
aps_teva = np.intersect1d(aps_test.columns, aps_val.columns)

#%%
#### TEST SET PREDICTIONS ####

## 111111111111111111111111111111111111111111111111111111111111111111111111111111111111 ##
cols_waps = train_waps.columns.tolist()
aps_test = aps_test[cols_waps]
aps_test = aps_test.replace(100, -110)

## Building prediction
pred_1_b = fit_knn1.predict(aps_test)
# Prediction results
unique1b, counts1b = np.unique(pred_1_b, return_counts=True)
dict(zip(unique1b, counts1b))

## Floor prediction
pred_1_f = fit_rf2.predict(aps_test)
# Prediction results
unique1f, counts1f = np.unique(pred_1_f, return_counts=True)
dict(zip(unique1f, counts1f))

## Longitude prediction
aps_test_pred = aps_test.copy()
aps_test_pred.loc[:, "BUILDINGID"] = pred_1_b
pred_1_lo = fit_xgb3.predict(aps_test_pred)

## Latitude prediction
pred_1_la = fit_rf4_2.predict(aps_test_pred)

# Long-Lat visualization
lo1 = pd.DataFrame(pred_1_lo)
la1 = pd.DataFrame(pred_1_la)
pred_lola = lo1.join(la1, lsuffix='LONGITUDE', rsuffix='LATITUDE')
f1 = pd.DataFrame(pred_1_f)
pred_lola = pred_lola.join(f1, rsuffix='FLOOR')

plot3d = plt.figure().gca(projection='3d')
plot3d.scatter(xs=pred_lola['0LONGITUDE'], ys=pred_lola['0LATITUDE'], zs=f1)
plot3d.set_zlabel('Floor')
plt.show()
plt.figure()

## 222222222222222222222222222222222222222222222222222222222222222222222222222222222222 ##
pred_2_b = fit_rf1.predict(aps_test)
pred_2_f = fit_rf2.predict(aps_test)

aps_test_pred = aps_test.copy()
aps_test_pred.loc[:, "BUILDINGID"] = pred_1_b

pred_xgb3 = fit_xgb3.predict(val_wapsb_pred)

val_wapsblo_pred = val_wapsb_pred.copy()
val_wapsblo_pred.loc[:, "LONGITUDE"] = pred_rf3.copy()

pred_rf4 = fit_rf4.predict(val_wapsblo_pred)

## 333333333333333333333333333333333333333333333333333333333333333333333333333333333333 ##

## 444444444444444444444444444444444444444444444444444444444444444444444444444444444444 ##


#%%
### PCA
# =============================================================================
# scaler=StandardScaler()  # instantiate
# scaler.fit()  # compute the mean and standard which will be used in the next command
# X_scaled=scaler.transform(cancer.data)# fit and transform can be applied together and I leave that for simple exercise
# # we can check the minimum and maximum of the scaled features which we expect to be 0 and 1
# print "after scaling minimum", X_scaled.min(axis=0)
# 
# pca=PCA(n_components=3) 
# pca.fit(X_scaled) 
# X_pca=pca.transform(X_scaled) 
# #let's check the shape of X_pca array
# print "shape of X_pca", X_pca.shape
# 
# ex_variance=np.var(X_pca,axis=0)
# ex_variance_ratio = ex_variance/np.sum(ex_variance)
# print ex_variance_ratio 
# =============================================================================
