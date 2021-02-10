#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:42:34 2020

Earthquake phase pick generator

@author: amt
"""
import uuid
from phase_pick_generator import get_generator, get_small_generator, get_phases_generator
import numpy as np
        
if __name__=="__main__":

    datatype='phases'
    if datatype=='small':
        my_data=get_small_generator()
        x=next(my_data) 
        count=0
        while count <= 1:
            print(count)
            y=np.zeros((100,500,11))
            for ii in range(100):
                x=next(my_data) 
                y[ii,:,:]=x
            count+=1     
            print(y.shape)
            np.save('associator_training_data/small_'+str(uuid.uuid4()),y)
    elif datatype=='phases':
        my_data=get_phases_generator()
        x=next(my_data) 
        count=0
        while count <= 218:
            print(count)
            y=np.zeros((100,500,12))
            for ii in range(100):
                x=next(my_data) 
                y[ii,:,:]=x
            count+=1     
            print(y.shape)
            np.save('associator_training_data/phases_'+str(uuid.uuid4()),y)        
    else:
        my_data=get_generator()
        x=next(my_data) 
        print(x.shape)
        count=0
        while count <= 1500:
            print(count)
            x=next(my_data) 
            count+=1     
            np.save('associator_training_data/orig_'+str(uuid.uuid4()),x)