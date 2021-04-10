#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:09:56 2020

Loads all data and separates into train, test, and develop

@author: searcy
"""

from glob import glob
import numpy as np
import matplotlib.pyplot as plt

datatype='master'
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
elif datatype=='master':
    data_files=glob('/Users/amt/Documents/cascadia_data_mining/cnn/associator_training_data/master*.npy')
    data=[np.load(f) for f in data_files]
    data_all=np.concatenate(data,axis=0)
    print(data_all.shape)
    np.save('master_comb_data.npy',data_all)
    
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
            
    np.save('master_train.npy',np.array(train))
    np.save('master_develop.npy',np.array(develop))
    np.save('master_test.npy',np.array(test))
    np.save('phases_train.npy',np.array(train))
    np.save('phases_develop.npy',np.array(develop))
    np.save('phases_test.npy',np.array(test))
    np.save('small_train.npy',np.array(train))
    np.save('small_develop.npy',np.array(develop))
    np.save('small_test.npy',np.array(test))
    np.save('phases_comb_data.npy',data_all[:, :, np.r_[0:5,10:12]])
    np.save('small_comb_data.npy',data_all[:, :, np.r_[0:10,11:12]])
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

# plot percentage of earthquakes
tmp=np.zeros((30000))
for ii in range(data_all.shape[0]):
    print(len(np.where(data_all[ii,:,11]==1)[0])/500)
    tmp[ii]=len(np.where(data_all[ii,:,11]==1)[0])/500
plt.hist(tmp)