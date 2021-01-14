#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:09:56 2020

Splits train, test, and develop datasets into inputs and outputs

@author: searcy
"""

import numpy as np

all_data=np.load('comb_data.npy')
train=np.load('train.npy') 
test=np.load('test.npy')
develop=np.load('develop.npy')


x_train=all_data[train,:,:5]
x_test=all_data[test,:,:5]
x_develop=all_data[develop,:,:5]

y_train=[all_data[train,:,5:-1],all_data[train,:,-1:]]
y_test=[all_data[test,:,5:-1],all_data[test,:,-1:]]
y_develop=[all_data[develop,:,5:-1],all_data[develop,:,-1:]]
