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

data_files=glob('associator_training_data/*.npy')
data=[np.load(f) for f in data_files]
data_all=np.concatenate(data,axis=0)
print(data_all.shape)
np.save('comb_data.npy',data_all)

test=[]
train=[]
develop=[]
for i,v in enumerate(data):
    _throw=np.random.uniform()
    if _throw>.9:
        test.append(i)
    elif _throw>.8:
        develop.append(i)
    else:
        train.append(i)

np.save('train.npy',np.array(train))
np.save('develop.npy',np.array(develop))
np.save('test.npy',np.array(test))