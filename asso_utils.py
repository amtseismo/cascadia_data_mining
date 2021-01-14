#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:09:56 2020

Does some input data normalizations

@author: searcy
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import pdb
import uuid


def fix_x_batch(x):
    # Station longitude, station latitude, elevation, phase, pick time
    minlat, maxlat, minlon, maxlon = 39.5067, 50.7067, -130.7661, -117.05375
    x[:,:,0]=(x[:,:,0]-minlon)/(maxlon-minlon) # normalizes lon between 0 and 1
    x[:,:,1]=(x[:,:,1]-minlat)/(maxlat-minlat) # normalizes lat between 0 and 1
    x[:,:,2]=x[:,:,2]/1000. # normalizes depth between 0 and 1
    x[:,:,4]=x[:,:,4]
    
    return x
  
def fix_y_batch(reg,lab):
    # Source Longitude, Source Latitude, Source Depth, Source Magnitude, Travel Time
    minlat, maxlat, minlon, maxlon = 39.5067, 50.7067, -130.7661, -117.05375
    reg[:,:,0]=(reg[:,:,0]-minlon)/(maxlon-minlon)
    reg[:,:,1]=(reg[:,:,1]-minlat)/(maxlat-minlat)
    reg[:,:,2]=reg[:,:,2]/100 
    reg[:,:,4]=reg[:,:,4]
    mask=lab[:,:,0]==0
    reg[mask]=0
    
    return reg,lab
    
def decode_y(y):
    minlat, maxlat, minlon, maxlon = 39.5067, 50.7067, -130.7661, -117.05375
    reg,classifier=y
    reg[:,:,0]=reg[:,:,0]*(maxlon-minlon)+minlon
    reg[:,:,1]=reg[:,:,1]*(maxlat-minlat)+minlat
    reg[:,:,2]=reg[:,:,2]*100
    reg[:,:,4]=reg[:,:,4]
    return reg
