#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IN PROGRESS
Purpose: Descriptiive visualization of UJIIndoorLoc dataset


Next step: Take out WAPs not in validation set, and those
with no observations


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

# %% Read and format data

df = pd.read_csv('data/trainingData.csv')
df_test = pd.read_csv('data/validationData.csv')

# %% Which WAPs are different between the two data sets?

# Keep columns that correspond to WAPs.
wap_names = [col for col in df if col.startswith('WAP')]
df = df[wap_names]
df_test = df_test[wap_names]

# Set all 100 values to NaN
df_na = df.replace(100, np.NaN)
df_test_na = df_test.replace(100, np.NaN)

na_col_train = df_na.isna().sum()
na_col_test = df_test_na.isna().sum()

# List of WAPs will all null values in train set
null_train = na_col_train[na_col_train == len(df_na)].index.tolist()
null_test = na_col_test[na_col_test == len(df_test_na)].index.tolist()

null_both = list(set(null_train).intersection(null_test))

print('There are', len(null_train), 'WAPs missing from the train set.')
print('There are', len(null_test), 'WAPs missing from the test set.')
print('There are', len(null_both), 'WAPs missing from both sets.')

# %% Density plots and basic visualizations


# Put all signals in the same column
df_m = pd.melt(df, value_vars = wap_names)


# Remove all 100 values
df_m = df_m[df_m.value != 100]

print(min(df_m.value))
print(max(df_m.value))

# Plot total histogram with log y axis
data = [go.Histogram(x=df_m['value'])]

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
df_wap['hover'] =  df_wap.index.map(str) + ', ' + df_wap.obs.map(str) + ' observations'

# WAP123
# 43 observations

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


