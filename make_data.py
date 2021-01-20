#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:42:34 2020

Earthquake phase pick generator

@author: amt
"""
import uuid
from phase_pick_generator import get_generator
import numpy as np
        
if __name__=="__main__":

    my_data=get_generator()
    x=next(my_data) 
    count=0
    while True:
        print(count)
        x=next(my_data) 
        count+=1     
        np.save('associator_training_data/'+str(uuid.uuid4()),x)