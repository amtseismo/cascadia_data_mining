#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:42:34 2020

Earthquake phase pick generator

@author: amt
"""
import uuid
from phase_pick_generator import get_generator, get_small_generator, get_phases_generator, get_master_generator, get_eqcount_generator
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
        
if __name__=="__main__":
    
    def make_small_data():
        my_data=get_small_generator()
        x=next(my_data) 
        y=np.zeros((100,500,11))
        for ii in range(100):
            x=next(my_data) 
            y[ii,:,:]=x   
        np.save('associator_training_data/small_'+str(uuid.uuid4()),y)
        return ()
        
    def make_phases_data():
        my_data=get_phases_generator()
        x=next(my_data) 
        y=np.zeros((100,500,7))
        for ii in range(100):
            x=next(my_data) 
            y[ii,:,:]=x
        np.save('associator_training_data/phases_'+str(uuid.uuid4()),y)  
        return ()
    
    def make_master_data():
        my_data=get_master_generator()
        x=next(my_data) 
        y=np.zeros((100,500,12))
        for ii in range(100):
            x=next(my_data) 
            y[ii,:,:]=x
        np.save('associator_training_data/master_'+str(uuid.uuid4()),y)  
        return ()
    
    def make_orig_data():
        my_data=get_generator()
        x=next(my_data)    
        np.save('associator_training_data/orig_'+str(uuid.uuid4()),x)        
        return ()
    
    def make_eqcount_data():
        my_data=get_eqcount_generator()
        x=next(my_data) 
        y=np.zeros((100,500,6))
        for ii in range(100):
            x=next(my_data) 
            y[ii,:,:]=x
        np.save('associator_training_data/eqcount_'+str(uuid.uuid4()),y)  
        return ()

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(make_eqcount_data)() for i in range(260))