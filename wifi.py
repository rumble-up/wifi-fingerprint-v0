#%%
## Importing Packages
import numpy as np  # for calculations
import pandas as pd  # for dataframes
import sklearn  # for caret
# import statsmodel.api as sm  # DOESN'T WORK
import matplotlib.pyplot as plt
plt.close('all')
from mpl_toolkits.mplot3d import Axes3D 
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn
import os, sys
import csv

import pickle
#%%
## Get and change working directory
os.getcwd()
os.chdir('/Users/denizminican/Dropbox/03-Data_and_Coding/Ubiqum/Repositories/wifi-fingerprint')

#%%
## Create the df
train = pd.read_csv('data/trainingData.csv')

## Change Time to DateTime
# =============================================================================
# train['TIMESTAMP'] = pd.to_datetime(train['TIMESTAMP'])
# =============================================================================


#%%
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
users = np.unique(train["USERID"])
train.plot.scatter(x= "LATITUDE", y= "LONGITUDE", c= "USERID")  # how to put colors instead of grayscale

#%%
## PLot 3D with floors
tdplot = plt.figure().gca(projection='3d')
tdplot.scatter(xs=train["LATITUDE"], ys=train["LONGITUDE"], zs=train["FLOOR"], c=train["FLOOR"])
tdplot.set_zlabel('Floor')
plt.show()
plt.figure()

#%%
## Plotly plots


#%%
## PhoneID dBm boxplots
train_phone = train.drop(['LATITUDE', 'LONGITUDE', 'USERID', 'FLOOR', 'BUILDINGID', 'SPACEID',
                          'RELATIVEPOSITION', 'TIMESTAMP'], axis = 1)

melted_phone = pd.melt(train_phone, id_vars=['PHONEID'], var_name='WAP')
melted_phone = melted_phone[melted_phone['value'] != 100]
melted_phone.boxplot(by = "PHONEID", column = "value")

#%%
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

with open('data/file_wap01.pkl', 'wb') as f:
    pickle.dump(wap01, f)

#%%
# Frequency of the WAPs
wapcount = melted_table['WAP'].value_counts()
wapcount.plot.hist()

py.iplot(go.Histogram(), filename='wap frequency histogram')  # ValueError: The first argument to the plotly.graph_objs.Histogram constructor must be a dict or an instance of plotly.graph_objs.Histogram 


#%%
## Check unique train/validation points
# Load validation set
val = pd.read_csv('data/validationData.csv')
# Melt for long format
melted_val = pd.melt(val, id_vars=['PHONEID', 'LATITUDE', 'LONGITUDE', 
                                       'FLOOR', 'BUILDINGID', 
                                       'TIMESTAMP'], var_name='WAP')
# Remove unused WAPs
melted_val = melted_val[melted_val['value'] != 100]

tra_wap = melted_table['WAP'].unique().tolist()  # 465, correct
val_wap = melted_val['WAP'].unique().tolist()  # Why do i have 370 not 367??

# Finding the common WAPs
common_waps = list(set(tra_wap) & set(val_wap))  # 312, correct

# Finding the common location points
tra_loc = melted_table.drop_duplicates(subset= ['LATITUDE', 'LONGITUDE'])  # ask if correct
tra_loc = tra_loc.drop(['SPACEID', 'RELATIVEPOSITION', 'USERID'], axis = 1)
val_loc = melted_val.drop_duplicates(subset= ['LATITUDE', 'LONGITUDE'])

#%%
# Joining them for plotting in same plot
tra_loc['dataset'] = 'training'
val_loc['dataset'] = 'validation'
all_loc = pd.concat([tra_loc, val_loc])

tra_loc.plot.scatter(x= "LATITUDE", y= "LONGITUDE")
val_loc.plot.scatter(x= "LATITUDE", y= "LONGITUDE")

#%%
# Training and validation unique points plot
fg = seaborn.FacetGrid(data=all_loc, hue='dataset')
fg.map(plt.scatter, 'LATITUDE', 'LONGITUDE').add_legend()

#%%
# 