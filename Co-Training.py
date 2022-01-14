#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import peakutils as pu
import pandas as pd
import numpy as np
import warnings
import pickle
import heapq
import json
import copy
import tqdm
import sys
import os
import time

from sklearn.linear_model import LinearRegression, LogisticRegression
from datetime import datetime, timedelta, date
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.utils import shuffle
from multiprocessing import Pool
from tqdm import tqdm, trange
from scipy.fft import ifft
from scipy import signal

import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from tensorflow.keras.utils import to_categorical  

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split


# In[ ]:

start_time = time.time()
window_size_features = 5
runs = 5
k = 100
U_prime_size = 500

DATA = 0

REBUILD = False

# IM - Individual Model
# GM - Group Model
MODEL_TYPE = 'GM'

# OS - Over Sample
# US - Under Sample
# NA - None
SAMPLE_TYPE = 'NA'

# 0 - Neural Network
# 1 - KNN
# 2 - Random Forest
# 3 - GaussianNB
# 4 - AdaBoostClassifier

model_to_use = 2

testing_range = [x/100 for x in np.arange(1, 5, 0.1)] + [x/100 for x in np.arange(5, 95,1)] + [x/100 for x in np.arange(95,99.7,0.1)]
subjects = [ "S" + str(i) for i in range(2, 18) if i != 12]


# In[ ]:





# In[ ]:


def baseline_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


# In[ ]:


# 0 - Neural Network
# 1 - KNN
# 2 - Random Forrest
# 3 - GaussianNB
# 4 - AdaBoostClassifier

models_dict = {
    '0': 'Deep Neural Network',
    '1': 'k-Nearest Neighbors', 
    '2': 'Random Forest',
    '3': 'Gaussian Naive Bayes',
    '4': 'AdaBoost'
}

model_to_use = 2


# In[ ]:


def predict_from_multiple_estimator(estimators, X_list, model_to_use, weights = None):

    if model_to_use == 0:
        pred1 = np.asarray([clf.predict(X) for clf, X in zip(estimators, X_list)])
    else:
        pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])
    pred2 = np.average(pred1, axis=0, weights=weights)
    pred = np.argmax(pred2, axis=1)

    # Convert integer predictions to original labels:
    return np.round(pred)


# In[ ]:


def get_data_with_gm_or_im_data(test_subject, MODEL_TYPE):
    
    if MODEL_TYPE == 'GM':
        data = pd.concat([all_features[subject] for subject in all_features.keys() if subject != test_subject])
        X_train = data.loc[:, data.columns != 'Label']
        y_train = data['Label']
        X_test = all_features[test_subject].loc[:, data.columns != 'Label']
        y_test = all_features[test_subject]['Label']
    elif MODEL_TYPE == 'IM':
        X_train, X_test, y_train, y_test = train_test_split(all_features[test_subject].loc[:, all_features[test_subject].columns != 'Label'], all_features[test_subject]['Label'], test_size=0.20)
    
    return X_train, X_test, y_train, y_test


# In[ ]:


def filer_and_clean_data(X_train, X_test):

    filter_col = [col for col in X_train if 'BVP' in col or 'ECG' in col or 'SCR' in col or 'SCL' in col or 'ACC' in col]
    X_train = X_train[filter_col]
    
    view_1_ind = [list(X_train.keys()).index(i) for i in X_train.keys() if 'BVP' in i or 'ECG' in i]
    view_2_ind = [list(X_train.keys()).index(i) for i in X_train.keys() if 'SCR' in i or 'SCL' in i or 'ACC' in i]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = X_test[filter_col]
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, view_1_ind, view_2_ind


# In[ ]:


def get_model(all_models=True):
    
    if all_models:
        if model_to_use == 0: model_1, model_2, model_3, model_4 = baseline_model(21), baseline_model(25), baseline_model(46), baseline_model(46)
        elif model_to_use == 1: model_1, model_2, model_3, model_4 = KNeighborsClassifier(n_neighbors=4), KNeighborsClassifier(n_neighbors=4), KNeighborsClassifier(n_neighbors=4), KNeighborsClassifier(n_neighbors=4)
        elif model_to_use == 2: model_1, model_2, model_3, model_4 = RandomForestClassifier(n_estimators=100), RandomForestClassifier(n_estimators=100), RandomForestClassifier(n_estimators=100), RandomForestClassifier(n_estimators=100)
        elif model_to_use == 3: model_1, model_2, model_3, model_4 = GaussianNB(), GaussianNB(), GaussianNB(), GaussianNB()
        elif model_to_use == 4: model_1, model_2, model_3, model_4 = AdaBoostClassifier(), AdaBoostClassifier(), AdaBoostClassifier(), AdaBoostClassifier() 

        return model_1, model_2, model_3, model_4
    else:
        if model_to_use == 0: model = baseline_model(46)
        elif model_to_use == 1: model = KNeighborsClassifier(n_neighbors=4)
        elif model_to_use == 2: model = RandomForestClassifier(n_estimators=100)
        elif model_to_use == 3: model = GaussianNB()
        elif model_to_use == 4: model = AdaBoostClassifier()

        return model


# In[ ]:


def update_labels(model_1, model_2, Unlabeled_X_train_half_1, Unlabeled_X_train_half_2, Labeled_X_train, Labeled_y_train, Unlabeled_X_train):
    vals_to_del = []
    
    if model_to_use == 0:
        predict_1 = model_1.predict(Unlabeled_X_train_half_1)
        predict_2 = model_2.predict(Unlabeled_X_train_half_2)
    else:
        predict_1 = model_1.predict_proba(Unlabeled_X_train_half_1)
        predict_2 = model_2.predict_proba(Unlabeled_X_train_half_2)
        
        if len(predict_1[0]) == 1:
            predict_1 = model_1.predict(Unlabeled_X_train_half_1)
        if len(predict_2[0]) == 1:
            predict_2 = model_2.predict(Unlabeled_X_train_half_2)
        
    if predict_1.ndim == 2:
        best_values_1 = list(set(np.concatenate((predict_1[:,0].argsort()[-k:][::-1], predict_1[:,1].argsort()[-k:][::-1]))))
    else:
        best_values_1 = list(set(list(predict_1.argsort()[-k:][::-1]) + list((predict_1 * -1).argsort()[-k:][::-1])))

    if predict_2.ndim == 2:
        best_values_2 = list(set(np.concatenate((predict_2[:,0].argsort()[-k:][::-1], predict_2[:,1].argsort()[-k:][::-1]))))
    else:
        best_values_2 = list(set(list(predict_2.argsort()[-k:][::-1]) + list((predict_2 * -1).argsort()[-k:][::-1])))
        
    for guess in best_values_1:
        Labeled_X_train = np.vstack([Labeled_X_train, np.append(Unlabeled_X_train_half_1[guess], Unlabeled_X_train_half_2[guess])])
        if model_to_use == 0:
            if predict_1.ndim == 2:
                Labeled_y_train = np.vstack([Labeled_y_train, to_categorical(np.argmax(predict_1[guess]), num_classes=2)])
            else:
                Labeled_y_train = np.vstack([Labeled_y_train, to_categorical(np.round(predict_1[guess]), num_classes=2)])
        else:
            if predict_1.ndim == 2:
                Labeled_y_train = np.append(Labeled_y_train, np.argmax(predict_1[guess]))
            else:
                Labeled_y_train = np.append(Labeled_y_train, np.round(predict_1[guess]))
        
    for guess in best_values_2:
        if guess in best_values_1:
            continue
        Labeled_X_train = np.vstack([Labeled_X_train, np.append(Unlabeled_X_train_half_1[guess], Unlabeled_X_train_half_2[guess])])
        if model_to_use == 0:
            if predict_2.ndim == 2:
                Labeled_y_train = np.vstack([Labeled_y_train, to_categorical(np.argmax(predict_2[guess]), num_classes=2)])
            else:
                Labeled_y_train = np.vstack([Labeled_y_train, to_categorical(np.round(predict_2[guess]), num_classes=2)])
        else:
            if predict_2.ndim == 2:
                Labeled_y_train = np.append(Labeled_y_train, np.argmax(predict_2[guess]))
            else:
                Labeled_y_train = np.append(Labeled_y_train, np.round(predict_2[guess]))

    Unlabeled_X_train_half_1 = np.delete(Unlabeled_X_train_half_1, best_values_1, axis=0)
    Unlabeled_X_train_half_2 = np.delete(Unlabeled_X_train_half_2, best_values_2, axis=0)
    Unlabeled_X_train = np.delete(Unlabeled_X_train, list(set(best_values_1 + best_values_2)), axis=0)
    return Unlabeled_X_train_half_1, Unlabeled_X_train_half_2, Unlabeled_X_train, Labeled_X_train, Labeled_y_train, list(set(best_values_1 + best_values_2))


# In[ ]:


def get_predictions(model_1, model_2, model_3, model_4, X_test, view_1_ind, view_2_ind, model_to_use):
    
    predict_1 = model_1.predict(X_test[:,view_1_ind])
    predict_2 = model_2.predict(X_test[:,view_2_ind])

    if model_to_use == 0:
        predict_combined = to_categorical(predict_from_multiple_estimator([model_1, model_2], [X_test[:,view_1_ind], X_test[:,view_2_ind]], model_to_use), num_classes = 2)
    else:
        predict_combined = predict_from_multiple_estimator([model_1, model_2], [X_test[:,view_1_ind], X_test[:,view_2_ind]], model_to_use)

    predict_1 = np.round(predict_1).astype(int)
    predict_2 = np.round(predict_2).astype(int)

    predict_supervised = np.round(model_3.predict(X_test)).astype(int)
    predict_all_supervised = np.round(model_4.predict(X_test)).astype(int)
    
    return predict_1, predict_2, predict_supervised, predict_all_supervised, predict_combined


# In[ ]:


def update_stats(model_to_use, test_subject, num_data, run, y_test, Labeled_y_train, all_updates=True, predict_updates=None, predict_repeat=None, predict_1=None, predict_2=None, predict_combined=None, predict_supervised=None, predict_all_supervised=None):
    
    if all_updates:
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['acc']['C1'].append(accuracy_score(y_test, predict_1))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['acc']['C2'].append(accuracy_score(y_test, predict_2))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['acc']['Combined'].append(accuracy_score(y_test, predict_combined))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['acc']['Supervised_partial'].append(accuracy_score(y_test, predict_supervised))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['acc']['Supervised_full'].append(accuracy_score(y_test, predict_all_supervised))

        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['roc']['C1'].append(roc_auc_score(y_test, predict_1))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['roc']['C2'].append(roc_auc_score(y_test, predict_2))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['roc']['Combined'].append(roc_auc_score(y_test, predict_combined))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['roc']['Supervised_partial'].append(roc_auc_score(y_test, predict_supervised))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['roc']['Supervised_full'].append(roc_auc_score(y_test, predict_all_supervised))
        
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['f1']['C1'].append(f1_score(y_test, predict_1, average='weighted', zero_division=0))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['f1']['C2'].append(f1_score(y_test, predict_2, average='weighted', zero_division=0))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['f1']['Combined'].append(f1_score(y_test, predict_combined, average='weighted', zero_division=0))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['f1']['Supervised_partial'].append(f1_score(y_test, predict_supervised, average='weighted', zero_division=0))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['f1']['Supervised_full'].append(f1_score(y_test, predict_all_supervised, average='weighted', zero_division=0))
    else:
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['acc']['Supervised_updates'].append(accuracy_score(y_test, predict_updates))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['roc']['Supervised_updates'].append(roc_auc_score(y_test, predict_updates))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['f1']['Supervised_updates'].append(f1_score(y_test, predict_updates, average='weighted', zero_division=0))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['acc']['Supervised_repeat'].append(roc_auc_score(y_test, predict_repeat))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['roc']['Supervised_repeat'].append(roc_auc_score(y_test, predict_repeat))
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['f1']['Supervised_repeat'].append(f1_score(y_test, predict_repeat, average='weighted', zero_division=0))
        
        all_data[str(model_to_use)][test_subject][str(num_data)][str(run)]['data_points'].append(len(Labeled_y_train))


# In[ ]:


t_sub = sys.argv[1]
# t_sub = 'S2'


# In[ ]:


def get_stats_dict():
    all_data = {
        str(model_to_use): {
            t_sub: {
                str(num_data): {
                    str(run): {
                        'data_points': [],
                        'acc': {
                            'C1': [],
                            'C2': [],
                            'Combined': [],
                            'Supervised_partial': [],
                            'Supervised_full': [],
                            'Supervised_updates': [],
                            'Supervised_repeat': []
                        }, 
                        'roc': {
                            'C1': [],
                            'C2': [],
                            'Combined': [],
                            'Supervised_partial': [],
                            'Supervised_full': [],
                            'Supervised_updates': [],
                            'Supervised_repeat': []
                        },
                        'f1': {
                            'C1': [],
                            'C2': [],
                            'Combined': [],
                            'Supervised_partial': [],
                            'Supervised_full': [],
                            'Supervised_updates': [],
                            'Supervised_repeat': []
                        }
                    } for run in range(runs)
                } for num_data in testing_range
            }
        }
    }
    return all_data


# In[ ]:


all_data = get_stats_dict()


# In[ ]:





# In[ ]:


def run_one_subject(test_subject):
    print("Working on {0}".format(test_subject))

    X_train, X_test, y_train, y_test = get_data_with_gm_or_im_data(test_subject, MODEL_TYPE)
    X_train, X_test, view_1_ind, view_2_ind = filer_and_clean_data(X_train, X_test)
    
    if DATA == 1:
        y_train = np.array([0 if x in [1, 2, 3, 4] else 1 if x in [5, 6, 7, 8] else -1 for x in y_train])
        y_test = np.array([0 if x in [1, 2, 3, 4] else 1 if x in [5, 6, 7, 8] else -1 for x in y_test])

    if model_to_use == 0:
        y_train = to_categorical(y_train, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)

    for num_data in tqdm(testing_range):

        for run in range(runs):
            
            X_train, y_train = shuffle(X_train, y_train)
            
            num_of_each_label = int((num_data * len(y_train)) / 2)
            elements_to_choose = np.concatenate((np.random.choice(np.where(y_train == 0)[0], num_of_each_label), np.random.choice(np.where(y_train == 1)[0], num_of_each_label)))
            elements_for_unlabeled = [x if x not in elements_to_choose else None for x in range(len(y_train))]
            elements_for_unlabeled = np.array(list(filter(None, elements_for_unlabeled)))
            
            Labeled_X_train = X_train[elements_to_choose]
            if DATA == 0:
                Labeled_y_train = y_train.values[elements_to_choose]
            elif DATA == 1:
                Labeled_y_train = y_train[elements_to_choose]
            
            Labeled_X_train, Labeled_y_train = shuffle(Labeled_X_train, Labeled_y_train)
            
            Labeled_X_train_c = Labeled_X_train.copy()
            Labeled_y_train_c = Labeled_y_train.copy()

            Unlabeled_X_train = X_train[ elements_for_unlabeled  ]
            
            if DATA == 0:
                Unlabeled_y_train = y_train.values[ elements_for_unlabeled ]
            elif DATA == 1:
                Unlabeled_y_train = y_train[ elements_for_unlabeled ]

            for X in range(1):
                Labeled_X_train, Labeled_y_train = shuffle(Labeled_X_train, Labeled_y_train)
                model_1, model_2, model_3, model_4 = get_model()

                if model_to_use == 0:
                    model_3.fit(Labeled_X_train, Labeled_y_train, verbose = 0)
                    model_4.fit(X_train, y_train, verbose = 0)
                else:
                    model_3.fit(Labeled_X_train, Labeled_y_train)
                    model_4.fit(X_train, y_train)

                predict_updates = model_3.predict(X_test)
                update_stats(model_to_use, test_subject, num_data, run, y_test, Labeled_y_train, all_updates=False, predict_updates=predict_updates, predict_repeat=predict_updates)
            
            elements_for_u_prime = np.concatenate((np.random.choice(np.where(Unlabeled_y_train == 0)[0], int(U_prime_size / 2)), np.random.choice(np.where(Unlabeled_y_train == 1)[0], int(U_prime_size / 2))))
            
            U_prime = Unlabeled_X_train[elements_for_u_prime]
            U_prime_y = Unlabeled_y_train[elements_for_u_prime]
            
            Unlabeled_X_train = np.delete(Unlabeled_X_train, elements_for_u_prime, axis=0)
            Unlabeled_y_train = np.delete(Unlabeled_y_train, elements_for_u_prime, axis=0)
            
            while True:

                X_train_half_1 = Labeled_X_train[:,view_1_ind]
                X_train_half_2 = Labeled_X_train[:,view_2_ind]

                if model_to_use == 0:
                    model_1.fit(X_train_half_1, Labeled_y_train, verbose = 0)
                    model_2.fit(X_train_half_2, Labeled_y_train, verbose = 0)
                else:
                    model_1.fit(X_train_half_1, Labeled_y_train)
                    model_2.fit(X_train_half_2, Labeled_y_train)

                Unlabeled_X_train_half_1 = U_prime[:,view_1_ind]
                Unlabeled_X_train_half_2 = U_prime[:,view_2_ind]
                
                Unlabeled_X_train_half_1, Unlabeled_X_train_half_2, U_prime, Labeled_X_train, Labeled_y_train, Deleted_Index = update_labels(model_1, model_2, Unlabeled_X_train_half_1, Unlabeled_X_train_half_2, Labeled_X_train, Labeled_y_train, U_prime)
                
                elements_for_u_prime_new = None
                if Unlabeled_X_train.shape[0] > 0:
                    count_of_deleted_neg_labels = len(np.where(U_prime_y[Deleted_Index] == 0)[0])
                    count_of_deleted_pos_labels = len(np.where(U_prime_y[Deleted_Index] == 1)[0])
                    
                    try:
                        elements_for_u_prime_new = np.concatenate(( np.unique(  np.random.choice(np.where(Unlabeled_y_train == 0)[0], count_of_deleted_neg_labels, replace=False)   ) , np.unique(  np.random.choice(np.where(Unlabeled_y_train == 1)[0], count_of_deleted_pos_labels, replace=False))))
                        elements_for_u_prime = np.concatenate((elements_for_u_prime, elements_for_u_prime_new))
                    except:
                        if len(np.where(Unlabeled_y_train == 0)[0]) == 0 and not len(np.where(Unlabeled_y_train == 1)[0]) == 0:
                            try:
                                elements_for_u_prime_new = np.unique(  np.random.choice(  [x for x in range(Unlabeled_y_train.shape[0])]  , count_of_deleted_pos_labels + count_of_deleted_neg_labels, replace=False))
                            except:
                                elements_for_u_prime_new = np.unique(  np.random.choice([x for x in range(Unlabeled_y_train.shape[0])], count_of_deleted_pos_labels + count_of_deleted_neg_labels))
                            elements_for_u_prime = np.concatenate((elements_for_u_prime, elements_for_u_prime_new))
                        elif len(np.where(Unlabeled_y_train == 1)[0]) == 0 and not len(np.where(Unlabeled_y_train == 0)[0]) == 0:
                            try:
                                elements_for_u_prime_new = np.unique(  np.random.choice([x for x in range(Unlabeled_y_train.shape[0])], count_of_deleted_pos_labels + count_of_deleted_neg_labels, replace=False))
                            except:
                                elements_for_u_prime_new = np.unique(  np.random.choice([x for x in range(Unlabeled_y_train.shape[0])], count_of_deleted_pos_labels + count_of_deleted_neg_labels))
                            elements_for_u_prime = np.concatenate((elements_for_u_prime, elements_for_u_prime_new))
                        else:
                            elements_for_u_prime_new = np.concatenate(( np.unique(  np.random.choice(np.where(Unlabeled_y_train == 0)[0], count_of_deleted_neg_labels)   ) , np.unique(  np.random.choice(np.where(Unlabeled_y_train == 1)[0], count_of_deleted_pos_labels))))
                            elements_for_u_prime = np.concatenate((elements_for_u_prime, elements_for_u_prime_new))
                            

                    try:
                        U_prime = np.concatenate((Unlabeled_X_train[elements_for_u_prime_new], U_prime))
                        U_prime_y = np.concatenate((Unlabeled_y_train[elements_for_u_prime_new], U_prime_y))
                    except:
                        print(Unlabeled_X_train.shape)
                        print(Unlabeled_y_train.shape)
                        print(elements_for_u_prime_new)

                    Unlabeled_X_train = np.delete(Unlabeled_X_train, elements_for_u_prime_new, axis=0)
                    Unlabeled_y_train = np.delete(Unlabeled_y_train, elements_for_u_prime_new, axis=0)
                
                model = get_model(all_models=False)
                model.fit(Labeled_X_train, Labeled_y_train)
                predict_updates = model.predict(X_test)
                
                model = get_model(all_models=False)
                model.fit(Labeled_X_train_c, Labeled_y_train_c)
                predict_repeat = model.predict(X_test)
                update_stats(model_to_use, test_subject, num_data, run, y_test, Labeled_y_train, all_updates=False, predict_updates=predict_updates, predict_repeat=predict_repeat)

                if Unlabeled_y_train.shape[0] == 0 and len(U_prime) == 0: break
            predict_1, predict_2, predict_supervised, predict_all_supervised, predict_combined = get_predictions(model_1, model_2, model_3, model_4, X_test, view_1_ind, view_2_ind, model_to_use)
            update_stats(model_to_use, test_subject, num_data, run, y_test, Labeled_y_train, predict_1=predict_1, predict_2=predict_2, predict_combined=predict_combined, predict_supervised=predict_supervised, predict_all_supervised=predict_all_supervised)


# In[ ]:


all_features = {}
for subject in subjects:
    path = 'Data/WESAD/GeneratedData/features_wesad_{0}_{1}.p'.format(subject, window_size_features)
    with open(path, 'rb') as file:
        all_features[subject] = pd.DataFrame(pickle.load(file, encoding='latin1')).T


# In[ ]:


run_one_subject(t_sub)


# In[ ]:


with open('TestingResults/{0}/{1}/runs_{2}_k_{3}_subject_{4}.json'.format('WESAD', MODEL_TYPE, runs, k, t_sub), 'w') as fp:
    json.dump(all_data, fp)
    
end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# running_all_for_control = {
#     num_data: {
#         "F1": [],
#         "ACC": [],
#         "ROC": [],
#     }for num_data in testing_range
# }


# In[ ]:





# In[ ]:


# for num_data in testing_range:
    
#     for run in range(runs):
        
#         X_train, X_test, y_train, y_test = get_data_with_gm_or_im_data(np.random.choice(all_subjects), MODEL_TYPE)
#         X_train, X_test, view_1_ind, view_2_ind = filer_and_clean_data(X_train, X_test)

#         if DATA == 1:
#             y_train = np.array([0 if x in [1, 2, 3, 4] else 1 if x in [5, 6, 7, 8] else -1 for x in y_train])
#             y_test = np.array([0 if x in [1, 2, 3, 4] else 1 if x in [5, 6, 7, 8] else -1 for x in y_test])


#         X_train, y_train = shuffle(X_train, y_train)

#         num_of_each_label = int((num_data / 100 * len(y_train)) / 2)
#         elements_to_choose = np.concatenate((np.random.choice(np.where(y_train == 0)[0], num_of_each_label), np.random.choice(np.where(y_train == 1)[0], num_of_each_label)))
#         elements_for_unlabeled = [x if x not in elements_to_choose else None for x in range(len(y_train))]
#         elements_for_unlabeled = np.array(list(filter(None, elements_for_unlabeled)))

#         Labeled_X_train = X_train[elements_to_choose]
#         if DATA == 0:
#             Labeled_y_train = y_train.values[elements_to_choose]
#         elif DATA == 1:
#             Labeled_y_train = y_train[elements_to_choose]

#         Labeled_X_train, Labeled_y_train = shuffle(Labeled_X_train, Labeled_y_train)


#         clf = RandomForestClassifier(n_estimators=100)
#         clf.fit(Labeled_X_train, Labeled_y_train)
#         predict = clf.predict(X_test)
#         acc = accuracy_score(y_test, predict)
#         roc = roc_auc_score(y_test, predict)
#         f1 = f1_score(y_test, predict, average='weighted', zero_division=0)

#         running_all_for_control[num_data]["F1"].append(f1)
#         running_all_for_control[num_data]["ACC"].append(acc)        
#         running_all_for_control[num_data]["ROC"].append(roc)    
#     print(f"{num_data} done")


# In[ ]:


# with open('TestingResults/running_all_for_control.json', 'w') as fp:
#     json.dump(running_all_for_control, fp)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




