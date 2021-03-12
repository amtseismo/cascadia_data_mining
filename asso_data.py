#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:09:56 2020

Splits train, test, and develop datasets into inputs and outputs

@author: searcy
"""

import numpy as np

datatype='data/eqcount'

all_data=np.load(datatype+'_comb_data.npy')
train=np.load(datatype+'_train.npy') 
test=np.load(datatype+'_test.npy')
develop=np.load(datatype+'_develop.npy')

print(all_data.shape)
x_train=all_data[train,:,:5]
x_test=all_data[test,:,:5]
x_develop=all_data[develop,:,:5]

y_train=all_data[train,:,5:]
y_test=all_data[test,:,5:]
y_develop=all_data[develop,:,5:]
