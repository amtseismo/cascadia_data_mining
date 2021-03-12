import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import pdb
import uuid
from train_associator_vCM import fix_data
from asso_utils import fix_x_batch,fix_y_batch_nomag,decode_y_nomag
from asso_data import x_train,y_train,x_develop,y_develop
from train_associator_v2 import loss,lon_loss,lat_loss,depth_loss,time_loss

if __name__=='__main__':


    tag='vCM'
    model= tf.keras.models.load_model('cp/'+tag+'_best_model.h5',
                                      custom_objects={'loss': loss,
                                                      'lon_loss':lon_loss,
                                                      'lat_loss':lat_loss,
                                                      'depth_loss':depth_loss,
                                                      'time_loss':time_loss}
                                   ) 

    
    xt,yt=fix_data(x_train,y_train)

    xd,yd=fix_data(x_develop,y_develop)


    
    yd_p=model.predict(xd)
