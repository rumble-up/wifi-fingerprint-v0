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


# %% Read and format data

df = pd.read_csv('data/trainingData.csv')

# Keep columns that correspond to WAPs.
waps = [col for col in df if col.startswith('WAP')]
df = df[waps]

# Histogram of average signal from each WAP
avg = df.mean(axis = 0, skipna = True).sort_values()
data = [go.Histogram(x=avg)]
plot(data)
