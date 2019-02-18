#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:42:34 2019

@author: chief
"""

# Load the library with the iris dataset
from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Set random seed
np.random.seed(42)

# Create an object called iris with the iris data
iris = load_iris()

# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# View the top 5 rows
df.head()

# Add a new column with the species names, this is what we are going to try to predict
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
train, test = df[df['is_train']==True], df[df['is_train']==False]

print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

features = df.columns[:4]




from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown = 'ignore')
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.fit(X)

enc.categories_


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