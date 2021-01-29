#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:42:34 2020

Earthquake phase pick generator

@author: amt
"""
import uuid
from phase_pick_generator import get_generator, get_small_generator
import numpy as np
        
if __name__=="__main__":


    small=True
    if small:
        my_data=get_small_generator()
        x=next(my_data) 
        count=0
        while count <= 92:
            print(count)
            y=np.zeros((100,500,11))
            for ii in range(100):
                x=next(my_data) 
                y[ii,:,:]=x
            count+=1     
            np.save('associator_training_data/small_'+str(uuid.uuid4()),y)
    else:
        my_data=get_generator()
        x=next(my_data) 
        count=0
        while count <= 500:
            print(count)
            x=next(my_data) 
            count+=1     
            np.save('associator_training_data/'+str(uuid.uuid4()),x)
        
    