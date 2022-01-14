#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:




import sys
import pickle
import pandas as pd
import numpy as np

from RASLPhysio.Signals import EDA_SCR_SCL_Decompose
from datetime import datetime, timedelta, date


# In[ ]:


def Scale_Baseline(labaled_df):
    grouped = labaled_df.groupby('Label')
    scaled = labaled_df.loc[:, labaled_df.columns != 'Label'] - np.nanmean(grouped.get_group(-1).drop(['Label'], axis = 1), axis = 0)
    scaled['Label'] = labaled_df['Label']
    df = scaled[scaled['Label'] != -1]
    return df


# In[ ]:




def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


# In[ ]:




subjects = [ "S" + str(i) for i in range(2, 18) if i != 12]


# In[ ]:



use_e4 = True
use_resp = True
window_size_in_seconds = int(sys.argv[2])


# In[ ]:




path = "Data/WESAD/{0}/{0}.pkl"
all_data = {}
sampling_rates = {'BVP': 64, 'EDA': 4, 'TEMP': 4, 'ACC': 32, 'Label': 700}
column_names_wrist = {'BVP': ['Wrist_BVP'], 'EDA': ['Wrist_EDA'], 'TEMP': ['Wrist_TEMP'], "ACC": ['Wrist_ACC_X', 'Wrist_ACC_Y', 'Wrist_ACC_Z'], 'Label': ['Label']}
column_names_chest = {'EDA': ['Chest_EDA'], 'Temp': ['Chest_TEMP'], "ACC": ['Chest_ACC_X', 'Chest_ACC_Y', 'Chest_ACC_Z'], 'Resp': ['Chest_RESP'], 'ECG': ['Chest_ECG'], 'EMG': ['Chest_EMG'], 'Label': ['Label']}


# In[ ]:




subject = sys.argv[1]
with open(path.format(subject), 'rb') as file:
    subject_data = pickle.load(file, encoding='latin1')

df_data_type = {}
df = pd.DataFrame(subject_data['label'])
df.index = [(1 / sampling_rates['Label']) * i for i in range(len(df))]
df.index = pd.to_datetime(df.index, unit='s')
df_data_type['Label'] = df
df.columns = column_names_wrist['Label']

if use_e4:
    for data_type in subject_data['signal']['wrist'].keys():
        df = pd.DataFrame(subject_data['signal']['wrist'][data_type])
        df.index = [(1 / sampling_rates[data_type]) * i for i in range(len(df))]
        df.index = pd.to_datetime(df.index, unit='s')
        df.columns = column_names_wrist[data_type]
        df_data_type['wrist_' + data_type] = df
    
if use_resp:
    for data_type in subject_data['signal']['chest'].keys():
        df = pd.DataFrame(subject_data['signal']['chest'][data_type])
        df.index = [(1 / 700) * i for i in range(len(df))]
        df.index = pd.to_datetime(df.index, unit='s')
        df.columns = column_names_chest[data_type]
        df_data_type['chest_' + data_type] = df

combined = pd.concat([df_data_type[data_type] for data_type in df_data_type.keys()], axis=1)
combined_max = combined.resample('{0}s'.format(window_size_in_seconds)).last()['Label']
combined_mean = combined.resample('{0}s'.format(window_size_in_seconds)).mean().loc[:, combined.columns != 'Label']
combined = pd.concat([combined_mean, combined_max], axis=1)
scr, scl = EDA_SCR_SCL_Decompose(combined['Chest_EDA'].values)
combined['Chest_SCR'] = scr
combined['Chest_SCL'] = scl

combined = combined[(combined.Label != 0) & (combined.Label != 5) & (combined.Label != 6) & (combined.Label != 7)]
combined.Label[combined[combined.Label.isin([1])].index] = -1
combined.Label[combined[combined.Label.isin([2])].index] = 1
combined.Label[combined[combined.Label.isin([3, 4])].index] = 0

combined = Scale_Baseline(combined)

combined.to_pickle("Data/WESAD/GeneratedData/{0}_Seconds_{1}.pkl".format(subject, window_size_in_seconds))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




