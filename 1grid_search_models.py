#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Status: In Progress
Purpose: Grid search different machine learning models

Created on Wed Feb 20 16:44:05 2019

@author: Laura Stupin
"""
# %% Setup --------------------------------------------------------------------

# Change working directory to the folder where script is stored.
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

import pandas as pd

# %% Read in processed data

df = pd.read_csv('data/processed/df.csv')

