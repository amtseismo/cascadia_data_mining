#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:09:56 2020

Uses trained associator to make predictions

@author: searcy
"""

import numpy as np
import matplotlib.pyplot as plt
from asso_data import x_test,y_test
from associator_nomag import build_model
from asso_utils import fix_x_batch,decode_y_nomag
from sklearn.cluster import MeanShift,estimate_bandwidth

batch_size=1
model=build_model(batch_size)
model.load_weights('best_nomag.tf')
pred=model.predict(fix_x_batch(x_test),batch_size=batch_size)
reg,cl=pred
reg_d=decode_y_nomag(pred)
np.save('predictions/Station_Regression',reg_d)
np.save('predictions/Station_Prediction',cl)
