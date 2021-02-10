#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:09:56 2020

Loads all data and separates into train, test, and develop

@author: searcy
"""

from glob import glob
import numpy as np
import pickle

datatype='small'
if datatype=='phases':
    data_files=glob('/Users/amt/Documents/cascadia_data_mining/cnn/associator_training_data/phases*.npy')
    data=[np.load(f) for f in data_files]
    data_all=np.concatenate(data,axis=0)
    print(data_all.shape)
    np.save('phases_comb_data.npy',data_all)
    
    test=[]
    train=[]
    develop=[]
    for i,v in enumerate(data_all):
        _throw=np.random.uniform()
        if _throw>.9:
            test.append(i)
        elif _throw>.8:
            develop.append(i)
        else:
            train.append(i)
    
    np.save('phases_train.npy',np.array(train))
    np.save('phases_develop.npy',np.array(develop))
    np.save('phases_test.npy',np.array(test))
elif datatype=='small':
    data_files=glob('/Users/amt/Documents/cascadia_data_mining/cnn/associator_training_data/small*.npy')
    data=[np.load(f) for f in data_files]
    data_all=np.concatenate(data,axis=0)
    print(data_all.shape)
    np.save('small_comb_data.npy',data_all)
    
    test=[]
    train=[]
    develop=[]
    for i,v in enumerate(data_all):
        _throw=np.random.uniform()
        if _throw>.9:
            test.append(i)
        elif _throw>.8:
            develop.append(i)
        else:
            train.append(i)
    
    np.save('small_train.npy',np.array(train))
    np.save('small_develop.npy',np.array(develop))
    np.save('small_test.npy',np.array(test))
else:
    data_files=glob('/Users/amt/Documents/cascadia_data_mining/cnn/associator_training_data/orig*.npy')
    data=[np.load(f) for f in data_files]
    data_all=np.concatenate(data,axis=0)
    print(data_all.shape)
    np.save('orig_comb_data.npy',data_all)
    
    test=[]
    train=[]
    develop=[]
    for i,v in enumerate(data_all):
        _throw=np.random.uniform()
        if _throw>.9:
            test.append(i)
        elif _throw>.8:
            develop.append(i)
        else:
            train.append(i)
    
    np.save('orig_train.npy',np.array(train))
    np.save('orig_develop.npy',np.array(develop))
    np.save('orig_test.npy',np.array(test))   