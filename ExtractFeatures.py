#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import os
import pickle
import sys
import pandas as pd

from RASLPhysio import Features
from datetime import datetime, timedelta, date


# %%

# %%


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


# %%


subjects = [ "S" + str(i) for i in range(2, 18) if i != 12]
subject =  sys.argv[1]

window_size = int(sys.argv[2])
window_size_features = int(sys.argv[3])


# %%


path = 'Data/WESAD/GeneratedData/{0}_Seconds_{1}.pkl'.format(subject, window_size)
with open(path, 'rb') as file:
    data = pickle.load(file, encoding='latin1')


# %%


params = Features.Features_Params()
params['rfft'] = False


# %%


all_features = {}
date = data.index[0]
while date + timedelta(seconds=window_size_features) <= data.index[-1]:
    end = nearest(data.index, date + timedelta(seconds=window_size_features))
    if date == end:
        ind = data.index.tolist().index(date) + 1
        date = data.index[ind]
        end = nearest(data.index, date + timedelta(seconds=window_size_features))
    window = data.loc[date:end,:]
    label = window['Label'][-1]
    features = {}
    for key in window.keys():
        if key == 'Label' or key == 'EDA': continue
        feats = Features.Extract_Features(window[key].values, params)
        for key_specific in feats.keys():
            features[ "{0}_{1}".format(key, key_specific)] = feats[key_specific]
    features['Label'] = label
    all_features[date] = features
    date = end
    print(date)
    
# features_to_keep = [
#     "SCL_mean",
#     "SCR_mean",
#     "SCL_std",
#     "SCR_std",
#     "BVP_std",
#     "ACC_X_mean",
#     "ACC_Y_mean",
#     "ACC_Z_mean",
# ]

Features_df = pd.DataFrame(all_features).T
# Features_df = Features_df[Features_df.columns.intersection(features_to_keep)]

pickle.dump( all_features, open( "Data/WESAD/GeneratedData/features_wesad_{0}_{1}.p".format(subject, window_size_features), "wb" ) )


# %%

# %%




