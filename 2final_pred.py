#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 12:55:39 2019

@author: chief
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Status: IN PROGRESS
Purpose: Use raw signals to make model prediction

Created on Feb 22 2019
@author: Laura Stupin

"""
# %% Setup --------------------------------------------------------------------

# Change working directory to the folder where script is stored.
from os import chdir, getcwd
wd=getcwd()
chdir(wd)

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.externals import joblib


