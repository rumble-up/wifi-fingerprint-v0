#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:42:34 2019

@author: chief
"""

import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot

df_raw = pd.read_csv('data/trainingData.csv')

df_val = pd.read_csv('data/validationData.csv')
dfoo = df_raw #.sample(100)

# 10 random samples from each building
foo = dfoo.groupby(['BUILDINGID', 'FLOOR']).apply(lambda x: x.sample(10)).reset_index(drop=True)
foo = foo.reset_index(drop=True)

floor4 = df_raw[df_raw['FLOOR'] == 4]
df = df_val

trace1 = go.Scatter3d(
        x = df.LONGITUDE,
        y = df.LATITUDE,
        z = df.FLOOR,
        mode = 'markers',
        marker=dict(
                color=df.FLOOR)
)
        
trace2 = go.Scatter3d(
        x = df_raw.LONGITUDE,
        y = df_raw.LATITUDE,
        z = df_raw.FLOOR,
        mode = 'markers',
        marker=dict(
                color=df_raw.FLOOR,
                colorscale='Viridis'
        )
)

plot([trace1, trace2])