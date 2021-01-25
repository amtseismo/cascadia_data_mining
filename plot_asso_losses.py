#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 09:58:43 2021

@author: amt
"""

import numpy as np 
import matplotlib.pyplot as plt

tmp=np.load("losses.pk",allow_pickle=True)
plt.plot(tmp['loss'],label='loss')
plt.plot(tmp['val_loss'],label='val loss')
plt.legend()