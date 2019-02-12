#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Status: IN PROGRESS
Purpose: Descriptiive visualization of UJIIndoorLoc dataset


Created on Thu Feb  7 10:58:43 2019
@author: Laura Stupin
"""



# %% Setup

# Change working directory to the folder where script is stored.
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

import pandas as pd
# import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import  plot #iplot, download_plotlyjs, init_notebook_mode
import numpy as np
import pickle


# %% Read and format data -----------------------------------------------------

df = pd.read_csv('data/trainingData.csv')
df_val = pd.read_csv('data/validationData.csv')

# %% Which WAPs are different between the two data sets? ----------------------

# Keep columns that correspond to WAPs.
wap_names = [col for col in df if col.startswith('WAP')]
df = df[wap_names]
df_val = df_val[wap_names]

# Set all 100 values to NaN
df_na = df.replace(100, np.NaN)
df_val_na = df_val.replace(100, np.NaN)

na_col_train = df_na.isna().sum()
na_col_test = df_val_na.isna().sum()

# List of WAPs will all null values in train set
null_train = na_col_train[na_col_train == len(df_na)].index.tolist()
null_test = na_col_test[na_col_test == len(df_val_na)].index.tolist()


null_both = list(set(null_train).intersection(null_test))

print('There are', len(null_train), 'WAPs missing from the train set.')
print('There are', len(null_test), 'WAPs missing from the test set.')
print('There are', len(null_both), 'WAPs missing from both sets.')

# Start storing lists of different types of waps
wap_groups = dict(null_train = null_train,
                  null_test = null_test)

# %% Density plots and basic visualizations for test and train sets


# Put all signals in the same column
df_m = pd.melt(df, value_vars = wap_names)
df_m['dataset'] = 'train'
df_val_m = pd.melt(df_val, value_vars = wap_names)
df_val_m['dataset'] = 'test'

# Append both dataframes
df_both = df_m.append(df_val_m)

# Remove all 100 values
df_both = df_both[df_both['value'] != 100]
#df_val_m = df_val_m[df_val_m['value'] != 100]

# Add columns with mW value
df_both['mW'] = pow(10, df_both.value/10)


# Plot dBm total histogram with log y axis
trace1 = go.Histogram(x=df_both['value'][df_both['dataset'] == 'train'])
trace2 = go.Histogram(x=df_both['value'][df_both['dataset'] == 'test'])
data = [trace1, trace2]

layout = go.Layout(
        yaxis = dict(
                type='log',
                autorange = True
        )
)
fig = go.Figure(data=data, layout=layout)
plot(fig)
   



# %% Average value by WAP

# Average signal strength/WAP histogram
avg = df_na.mean(axis = 0, skipna = True).sort_values(ascending = False)
data = [go.Histogram(x=avg)]
plot(data, filename = 'plots/avg_sig_histogram.html')

# Count NAs per observation
na_count = df_na.isna().sum(axis =1)
# Plot WAPs per real observation
non_na = 520 - na_count
data = [go.Histogram(x=non_na)]
plot(data)


# Scatter plots
# Average signal by WAP
avg = avg.sort_index()
# Observations by WAP
obs = df_na.count()

# Dataframe of characteristics by WAP
df_wap = pd.DataFrame(dict(avg = avg, obs = obs))

# %% Investigate signals above -30 dBm training alone
# THERE ARE NO OUTLIERS IN TEST SET
df_both_outliers = df_both[df_both['value'] > -30]

# Store in WAP dictionary
wap_groups['above30dbm'] = df_both_outliers['variable'].unique().tolist()
outliers = pd.pivot_table(df_both_outliers, 
                          values = 'value', 
                          index = ['variable'],
                          aggfunc = 'count')


# Add information about outliers
outliers.columns = ['above30dBm']
df_wap = df_wap.join(outliers, how='outer') 
df_wap['above30_%'] = df_wap.above30dBm/df_wap.obs                          
                          
df_wap['hover'] =  df_wap.index.map(str) + ', ' + df_wap.obs.map(str) + ' observations'

#%% WAP percentage outlier scatter chart

out = df_wap[df_wap['above30_%'] > 0]

trace = go.Scatter(
        x=out.index,
        y=out['above30_%'],
        mode = 'markers'
)
data = [trace]
plot(data)

#%% Master WAP graph

trace = go.Scatter(
        x = df_wap.index,
        y = df_wap.avg, 
        text = df_wap.hover,
        hoverinfo = 'text',
        mode = 'markers',
        marker = dict(
                color = np.log(df_wap.obs),
                colorscale = 'Viridis',
                showscale = False
        )
)

layout = go.Layout(
        title='Characteristics of WAPs',
        yaxis=dict(
                title = 'Average signal strength'
        )
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
plot(fig, filename='plots/characteristics_of_WAPs.html')

# %% Pickling staging area
# Load an object in     
#with open('data/wap_buildings.pkl', 'rb') as f:
#    d = pickle.load(f)
#    
#wap_groups['b0'] = d['wap0']
#wap_groups['b01'] = d['wap01']
#wap_groups['b02'] = d['wap02']
#wap_groups['b1'] = d['wap1']
#wap_groups['b12'] = d['wap12']
#wap_groups['b2'] = d['wap2']

# Save an object in the environment
with open('data/wap_groups.pkl', 'wb') as f:
    pickle.dump(wap_groups, f)

# Load an object in     
with open('data/wap_groups.pkl', 'rb') as f:
    test = pickle.load(f)

#%% Sandbox and archive


# Plot mW on x-axis histogram 
# This plot does not seem useful
#trace1 = go.Histogram(x=df_m['mW'][df_m['dataset'] == 'train'])
#trace2 = go.Histogram(x=df_m['mW'][df_m['dataset'] == 'test'])
#data = [trace1] #, trace2]
#
#layout = go.Layout(
#        xaxis = dict(
#                type='log',
#                autorange = True
#        )
#)
#fig = go.Figure(data=data, layout=layout)
#plot(fig)     
    

