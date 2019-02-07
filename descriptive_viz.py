#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: Descriptiive visualization of UJIIndoorLoc dataset

Created on Thu Feb  7 10:58:43 2019
@author: Laura Stupin
"""

# %% Startup

# Change working directory to the folder where script is stored.
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

import pandas as pd
# import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import  plot, iplot #, download_plotlyjs, init_notebook_mode
import numpy as np

# %% Read and format data

df = pd.read_csv('data/trainingData.csv')

# Keep columns that correspond to WAPs.
waps = [col for col in df if col.startswith('WAP')]
df = df[waps]
# Put all signals in the same column
df_m = pd.melt(df, value_vars = waps)
len(df_m)

# Remove all 100 values
df_m = df_m[df_m.value != 100]

print(min(df_m.value))
print(max(df_m.value))
# Plot total histogram
data = [go.Histogram(x=df_m['value'])]
plot(data)


 

# %% Average value by WAP

# Histogram of average signal from each WAP
# Set all 100 values to NaN
df_na = df.replace(100, np.NaN)

# Plot average signal strength/WAP histogram
avg = df_na.mean(axis = 0, skipna = True).sort_values()
data = [go.Histogram(x=avg)]
plot(data, filename = 'plots/avg_sig_histogram.html')

# Count NAs per observation
na_count = df_na.isna().sum(axis =1)
# Plot WAPs per real observation
non_na = 520 - na_count
data = [go.Histogram(x=non_na)]
plot(data)
