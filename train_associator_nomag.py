#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:09:56 2020

Trains the associator

@author: searcy
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import pdb
import uuid
from associator_nomag import build_model
from asso_utils import fix_x_batch,fix_y_batch_nomag
from asso_data import x_train,y_train,x_develop,y_develop
import pickle

def get_valid_loss(model,x_develop,y_develop,batch_size=32):
    loss=[]
    for batch in range(0,len(x_develop) ,batch_size ):
        model.reset_states()
        if batch+batch_size > len(x_develop):
            pass
        else:            
            batch_x=fix_x_batch(x_train[batch:batch+batch_size,:,:].copy())
            batch_y=fix_y_batch_nomag(y_train[0][batch:batch+batch_size,:,:].copy(),
                                 y_train[1][batch:batch+batch_size,:,:].copy())            
            loss.append(model.test_on_batch(batch_x,batch_y)[0])
    model.reset_states()
    return np.mean(loss)
    
if __name__=="__main__":
    batch_size=32 # specify batch size
    model=build_model(batch_size) # build the model with that batch size
    length=x_train.shape[1] # gets batch length (10000)
    count=0
    start=0
    losses=[]
    epochs=0
    avail_index=list(range(x_train.shape[0])) # get total number of available batches
    print(x_train.shape) # print training data shape (895, 10000, 5)
    batch_i=np.random.choice(avail_index,batch_size,replace=False) # from the available weeks of synthetic data select a subset 
    [avail_index.remove(i) for i in batch_i] # remove the batches selected from the available weeks
    valid_losses=[]
    epoch_losses=[]
    best_loss=None
    print('Epoch '+str(epochs))
    while True:
        _bloss=[]
        end=np.random.choice(range(50,500))+start # This is used to pull only a small time slice of each batch
        if end >= length:
            end=length-1 
        # Station longitude, station latitude, elevation, phase, pick time
        batch_x=fix_x_batch(x_train[batch_i,start:end,:].copy()) # get inputs
        # Source longitude, latitude, depth, travel time
        batch_y=fix_y_batch_nomag(y_train[0][batch_i,start:end,:].copy(),
                              y_train[1][batch_i,start:end,:].copy()) # get outputs
        
        loss=model.train_on_batch(batch_x,batch_y) # run a single gradient update 
        _bloss.append(loss) # append the loss

        start=end
        model.reset_states() # clears network hidden states

        if start >= length -1: # if youve reached the end of the batch
            print('End of batch '+str(len(avail_index)//batch_size)+' steps left')
            losses.append(np.mean(_bloss,axis=0)) # save the mean of the batch loss
            for _n,_l in zip(model.metrics_names,np.mean(_bloss,axis=0)):
                print(_n,_l)
            _bloss=[] # reset batch loss

            if len(avail_index) <= batch_size: # if you dont have enough weeks for another full batch
                avail_index=list(range(x_train.shape[0])) # restart with the full set
                epochs+=1 # update the epoch
                print('Epoch '+str(epochs))
                valid_loss=get_valid_loss(model,x_develop,y_develop,batch_size) # get the validation loss on the develop data
                if best_loss==None or valid_loss < best_loss:
                    best_loss=valid_loss # if the loss is better than the best loss, save it and the model
                    print('New Best Loss',valid_loss)
                    model.save_weights('./best_nomag.tf')
                valid_losses.append(valid_loss) # append the validation loss for the epoch
                epoch_losses.append(np.mean(losses)) # append the loss for the epoch
                losses=[] # reset losses
                print('loss: ',epoch_losses[-1], ' valid: ',valid_loss)
                pickle.dump({"loss":epoch_losses,'val_loss':valid_losses}, open('losses.pk','wb') ) # save stuff
            batch_i=np.random.choice(avail_index,batch_size,replace=False) # make next training set
            [avail_index.remove(i) for i in batch_i]
            
            start=0

 
#                if count % 10 ==0:
#                    model.reset_states()
#                    _r,_c=model.predict(fix_x_batch(all_data[:batch_size,:,:5].copy()))
#                    print(_r,_c)                    
#                    model.save_weights('weights_'+str(count)+".tf")


            model.reset_states()
            count+=1
                
            
#         #heckpoint = tf.keras.callbacks.ModelCheckpoint('asso_best.tf', monitor='loss',save_weights_only=True, verbose=1, save_best_only=True, mode='min')
#         #    model.load_weights('asso_best.tf')   

#         #    model.load_weights('asso_best.tf')



#     # for i in range(10):
#     #     points=np.concatenate([np.expand_dims(p,0) for p in pred[i] if p[5]>0 ],axis=0)
#     #     events=np.concatenate([np.expand_dims(p,0) for p in y[i] if p[5]!=0 ],axis=0)
        
#     #     plt.scatter(points[:,0],points[:,1],c=points[:,4] )
#     #     plt.scatter(events[:,0],events[:,1],c=events[:,4],marker='x')
#     #     plt.show()