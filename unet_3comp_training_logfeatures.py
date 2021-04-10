#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:27:07 2020

Train a CNN to pick P and S wave arrivals with log features

@author: amt
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import h5py
from scipy import signal
import unet_tools
import argparse

# # parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("-subset", "--subset", help="train on a subset or no?", type=int)
# parser.add_argument("-pors", "--pors", help="train P or S network", type=int)
# parser.add_argument("-train", "--train", help="do you want to train?", type=int)
# parser.add_argument("-drop", "--drop", help="want to add a drop layer to the network", type=int)
# parser.add_argument("-plots", "--plots", help="want plots", type=int)
# parser.add_argument("-resume", "--resume", help="want to resume training?", type=int)
# parser.add_argument("-large", "--large", help="what size do you want the network to be?", type=float)
# parser.add_argument("-epochs", "--epochs", help="how many epochs", type=int)
# parser.add_argument("-std", "--std", help="standard deviation of target", type=float)
# parser.add_argument("-sr", "--sr", help="sample rate in hz", type=int)
# args = parser.parse_args()

# subset=args.subset #True # train on a subset or the full monty?
# ponly=args.pors #True # 1 - P+Noise, 2 - S+noise
# train=args.train #True # do you want to train?
# drop=args.drop #True # drop?
# plots=args.plots #False # do you want to make some plots?
# resume=args.resume #False # resume training
# large=args.large # large unet
# epos=args.epochs # how many epocs?
# std=args.std # how long do you want the gaussian STD to be?
# sr=args.sr

subset=0 #True # train on a subset or the full monty?
ponly=1 #True # 1 - P+Noise, 2 - S+noise
train=0 #True # do you want to train?
drop=1 #True # drop?
plots=1 #False # do you want to make some plots?
resume=0 #False # resume training
large=0.5 # large unet
epos=50# how many epocs?
std=0.05 # how long do you want the gaussian STD to be?
sr=100

epsilon=1e-6

print("subset "+str(subset))
print("ponly "+str(ponly))
print("train "+str(train))
print("drop "+str(drop))
print("plots "+str(plots))
print("resume "+str(resume))
print("large "+str(large))
print("epos "+str(epos))
print("std "+str(std))
print("sr "+str(sr))

# LOAD THE DATA
print("LOADING DATA")
if ponly:
    if subset:
        n_data = h5py.File('./detector_training_data/pnsn_ncedc_3comp_N_100_training_data_small.h5', 'r')
        x_data = h5py.File('./detector_training_data/pnsn_ncedc_3comp_P_100_training_data_small.h5', 'r')
        model_save_file="unet_3comp_logfeat_10000_pn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf" 
    else:
        n_data = h5py.File('./detector_training_data/pnsn_ncedc_3comp_N_100_training_data.h5', 'r')
        x_data = h5py.File('./detector_training_data/pnsn_ncedc_3comp_P_100_training_data.h5', 'r')
        model_save_file="unet_3comp_logfeat_250000_pn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf" 
else:
    if subset:
        n_data = h5py.File('./detector_training_data/pnsn_ncedc_3comp_N_100_training_data_small.h5', 'r')
        x_data = h5py.File('./detector_training_data/pnsn_ncedc_3comp_S_100_training_data_small.h5', 'r')
        model_save_file="unet_3comp_logfeat_10000_sn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf" 
    else:
        n_data = h5py.File('./detector_training_data/pnsn_ncedc_3comp_N_100_training_data.h5', 'r')
        x_data = h5py.File('./detector_training_data/pnsn_ncedc_3comp_S_100_training_data.h5', 'r')
        model_save_file="unet_3comp_logfeat_250000_sn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"  
        
if large:
    fac=large
    model_save_file="large_"+str(fac)+"_"+model_save_file

if drop:
    model_save_file="drop_"+model_save_file

# plot the data
if plots:
    # plot ps to check
    plt.figure()
    for ii in range(20):
        if ponly:
            ntmp=x_data['waves'][ii]
        else:
            ntmp=x_data['waves'][ii]
        plt.plot(ntmp/np.max(np.abs(ntmp))+ii)
        
    # plot noise to check
    plt.figure()
    for ii in range(20):
        ntmp=n_data['noise'][ii]
        plt.plot(ntmp/np.max(np.abs(ntmp))+ii)

# MAKE TRAINING AND TESTING DATA
print("MAKE TRAINING AND TESTING DATA")
np.random.seed(0)
if ponly:
    siginds=np.arange(x_data['waves'].shape[0])
    noiseinds=np.arange(n_data['noise'].shape[0])
else:
    siginds=np.arange(x_data['waves'].shape[0])
    noiseinds=np.arange(n_data['noise'].shape[0])
np.random.shuffle(siginds)
np.random.shuffle(noiseinds)
sig_train_inds=np.sort(siginds[:int(0.75*len(siginds))])
noise_train_inds=np.sort(noiseinds[:int(0.75*len(noiseinds))])
sig_test_inds=np.sort(siginds[int(0.75*len(siginds)):])
noise_test_inds=np.sort(noiseinds[int(0.75*len(noiseinds)):])

# # do the shifts and make batches
# print("SETTING UP GENERATOR")

# generate batch data
print("FIRST PASS WITH DATA GENERATOR")
my_data=unet_tools.my_3comp_data_generator(32,x_data,n_data,sig_train_inds,noise_train_inds,sr,std)
x,y=next(my_data)

# PLOT GENERATOR RESULTS
if plots:
    for ind in range(5):
        fig, (ax0,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True)
        t=1/sr*np.arange(x.shape[1])
        ax0.set_xlabel('Time (s)')
        ax0.set_ylabel('Amplitude', color='tab:red')
        ax0.plot(t, (np.exp(x[ind,:,0])-epsilon)*x[ind,:,1], color='tab:red', label='data')
        ax0.tick_params(axis='y')
        ax0.legend(loc="lower right")
        ax1 = ax0.twinx()  # instantiate a second axes that shares the same x-axis
        ax1.set_ylabel('Prediction', color='black')  # we already handled the x-label with ax1
        ax1.plot(t, y[ind,:], color='black', linestyle='--', label='target')
        ax1.legend(loc="upper right")
        ax2.plot(t, x[ind,:,0], color='tab:green', label='ln(data amp)')
        ax2.plot(t, x[ind,:,1], color='tab:blue', label='data sign')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax2.legend(loc="lower right")
        # plt.show()

# BUILD THE MODEL
print("BUILD THE MODEL")
if drop:
    model=unet_tools.make_large_unet_drop(fac,sr,ncomps=3)    
else:
    model=unet_tools.make_large_unet(fac,sr,ncomps=3)  
        
# ADD SOME CHECKPOINTS
print("ADDING CHECKPOINTS")
checkpoint_filepath = './checks/'+model_save_file+'_{epoch:04d}.ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_weights_only=True, verbose=1,
    monitor='val_acc', mode='max', save_best_only=True)

# TRAIN THE MODEL
print("TRAINING!!!")
if train:
    batch_size=32
    if resume:
        print('Resuming training results from '+model_save_file)
        model.load_weights(checkpoint_filepath)
    else:
        print('Training model and saving results to '+model_save_file)
        
    csv_logger = tf.keras.callbacks.CSVLogger(model_save_file+".csv", append=True)
    history=model.fit_generator(my_data_generator(batch_size,x_data,n_data,sig_train_inds,noise_train_inds,sr,std),
                        steps_per_epoch=(len(sig_train_inds)+len(noise_train_inds))//batch_size,
                        validation_data=my_data_generator(batch_size,x_data,n_data,sig_test_inds,noise_test_inds,sr,std),
                        validation_steps=(len(sig_test_inds)+len(noise_test_inds))//batch_size,
                        epochs=epos, callbacks=[model_checkpoint_callback,csv_logger])
    model.save_weights("./"+model_save_file)
else:
    print('Loading training results from '+model_save_file)
    model.load_weights("./result_files/"+model_save_file)
    
# plot the results
if plots:
    # training stats
    training_stats = np.genfromtxt("./result_files/"+model_save_file+'.csv', delimiter=',',skip_header=1)
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(training_stats[:,0],training_stats[:,1])
    ax1.plot(training_stats[:,0],training_stats[:,3])
    ax1.legend(('acc','val_acc'))
    ax2.plot(training_stats[:,0],training_stats[:,2])
    ax2.plot(training_stats[:,0],training_stats[:,4])
    ax2.legend(('loss','val_loss'))
    ax2.set_xlabel('Epoch')
    ax1.set_title(model_save_file)

# See how things went
my_test_data=my_data_generator(20,x_data,n_data,sig_test_inds,noise_test_inds,sr,std,valid=True)
x,y=next(my_test_data)

test_predictions=model.predict(x)

# PLOT A FEW EXAMPLES
if plots:
    for ind in range(15):
        fig, ax1 = plt.subplots()
        t=1/100*np.arange(x.shape[1])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        trace=np.multiply(np.power(x[ind,:,0],10),x[ind,:,1])
        ax1.plot(t, trace, color='tab:red') #, label='data')
        ax1.tick_params(axis='y')
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Prediction')  # we already handled the x-label with ax1
        ax2.plot(t, test_predictions[ind,:], color='tab:blue') #, label='prediction')
        ax2.plot(t, y[ind,:], color='black', linestyle='--') #, label='target')
        ax2.tick_params(axis='y')
        ax2.set_ylim((-0.1,2.1))
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend(('prediction','target'))
        plt.show()
