#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:09:56 2020

Calculates clustering parameters

@author: searcy
"""

import numpy as np
import matplotlib.pyplot as plt
from asso_data import x_test,y_test
from sklearn.cluster import MeanShift,estimate_bandwidth

regression=np.load('predictions/Station_Regression.npy')
prediction=np.load('predictions/Station_Prediction.npy')

#Regression and Prediction Errors
cl_truth=[]
cl_pred=[]
reg_res_good=[]
reg_res_bad=[]
threshold=0.5

diffs=np.zeros(y_test[0].shape)
baddiffs=np.zeros(y_test[0].shape)

for index in range(len(x_test)):
    # # model outputs
    # reg_d=regression[index:index+1]
    # cl_pred=prediction[index:index+1]
    # # true values
    # true_reg=y_test[0][index:index+1]
    # true_cl=y_test[1][index:index+1]
    # cl_truth+=list(true_cl.flatten())
       
    for i in range(regression.shape[1]):
        # if actually a true event and predicted to be true
        if y_test[1][index,i,0] and prediction[index,i,0]>threshold:
            diffs[index,i,:]=regression[index,i,:]-y_test[0][index,i,:] 
        # if actually a true event and not predicted to be true
        if y_test[1][index,i,0] and prediction[index,i,0]<=threshold:
            baddiffs[index,i,:]=regression[index,i,:]-y_test[0][index,i,:] 

#    mask=true_cl==1
#    reg_mask=np.tile(mask,[1,1,5])
   
#     reg_res_good+=[reg_d[0,i,:]-true_reg[0,i,:] for i in range(reg_d.shape[1]) if true_cl[0,i,0] and cl_pred[0,i,0]>threshold]
#     reg_res_bad+=[reg_d[0,i,:]-true_reg[0,i,:] for i in range(reg_d.shape[1]) if true_cl[0,i,0] and cl_pred[0,i,0]<threshold]

# reg_res_good=np.array(reg_res_good)
# reg_res_bad=np.array(reg_res_bad)

# reg_devs=[]
# for i in range(reg_res_good.shape[1]):
#     # max_val=max([np.max(reg_res_good[:,i]),np.max(reg_res_bad[:,i])])
#     # min_val=min([np.min(reg_res_good[:,i]),np.min(reg_res_bad[:,i])])
#     print(i,np.std(reg_res_good[:,i]))
#     reg_devs.append(np.std(reg_res_good[:,i]))
# np.save('predictions/Cluster_Devs',np.array(reg_devs))    
    
    
for batchid in range(1):
    # get wehre predictions are nonzero
    inds=np.where(diffs[batchid,:,0]!=0)[0]
    badinds=np.where(baddiffs[batchid,:,0]!=0)[0]
    # plot differences between prediction and true
    fig,ax=plt.subplots(2,3,figsize=(12,9))
    ax[0,0].hist(diffs[batchid,inds,0],bins=100,label='diffs')
    ax[0,0].hist(baddiffs[batchid,badinds,0],bins=100,label='bad diffs')
    ax[0,0].axvline(np.mean(diffs[batchid,inds,0]),color='r')
    ax[0,0].set_title('Pred-True Lons')
    ax[0,0].legend()
    #---------------
    ax[0,1].hist(diffs[batchid,inds,1],bins=100)
    ax[0,1].hist(baddiffs[batchid,badinds,1],bins=100)
    ax[0,1].axvline(np.mean(diffs[batchid,inds,1]),color='r')
    ax[0,1].set_title('Pred-True Lats')
    #---------------
    ax[0,2].hist(diffs[batchid,inds,2],bins=100)
    ax[0,2].hist(baddiffs[batchid,badinds,2],bins=100)
    ax[0,2].axvline(np.mean(diffs[batchid,inds,2]),color='r')
    ax[0,2].set_title('Pred-True Depths')
    #---------------
    ax[1,0].hist(diffs[batchid,inds,3],bins=100)
    ax[1,0].hist(baddiffs[batchid,badinds,3],bins=100)
    ax[1,0].axvline(np.mean(diffs[batchid,inds,3]),color='r')
    ax[1,0].set_title('Pred-True Mags')
    #---------------
    ax[1,1].hist(diffs[batchid,inds,4],bins=100)
    ax[1,1].hist(baddiffs[batchid,badinds,4],bins=100)
    ax[1,1].axvline(np.mean(diffs[batchid,inds,4]),color='r')
    ax[1,1].set_title('Pred-True Epoch')
    
    # plot predictions and true for all inds
    fig,ax=plt.subplots(2,3,figsize=(12,9))
    ax[0,0].plot(regression[batchid,inds,0],y_test[0][batchid,inds,0],'o')
    ax[0,0].set_title('Pred and True Lons')
    ax[0,0].set_xlabel('Pred')
    ax[0,0].set_ylabel('True')
    lims = [np.min([ax[0,0].get_xlim(), ax[0,0].get_ylim()]),  # min of both axes
        np.max([ax[0,0].get_xlim(), ax[0,0].get_ylim()])]  # max of both axes
    ax[0,0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    #---------------
    ax[0,1].plot(regression[batchid,inds,1],y_test[0][batchid,inds,1],'o')
    ax[0,1].set_title('Pred vs True Lats')
    ax[0,1].set_xlabel('Pred')
    ax[0,1].set_ylabel('True')
    lims = [np.min([ax[0,1].get_xlim(), ax[0,1].get_ylim()]),  # min of both axes
        np.max([ax[0,1].get_xlim(), ax[0,1].get_ylim()])]  # max of both axes
    ax[0,1].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    #---------------
    ax[0,2].plot(regression[batchid,inds,2],y_test[0][batchid,inds,2],'o')
    ax[0,2].set_title('Pred and True Depths')
    ax[0,2].set_xlabel('Pred')
    ax[0,2].set_ylabel('True')
    lims = [np.min([ax[0,2].get_xlim(), ax[0,2].get_ylim()]),  # min of both axes
        np.max([ax[0,2].get_xlim(), ax[0,2].get_ylim()])]  # max of both axes
    ax[0,2].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    #---------------
    ax[1,0].plot(regression[batchid,inds,3],y_test[0][batchid,inds,3],'o')
    ax[1,0].set_title('Pred and True Mags')
    ax[1,0].set_xlabel('Pred')
    ax[1,0].set_ylabel('True')
    lims = [np.min([ax[1,0].get_xlim(), ax[1,0].get_ylim()]),  # min of both axes
        np.max([ax[1,0].get_xlim(), ax[1,0].get_ylim()])]  # max of both axes
    ax[1,0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    #---------------
    ax[1,1].plot(regression[batchid,inds,4],y_test[0][batchid,inds,4],'o')
    ax[1,1].set_title('Pred and True Epoch')
    ax[1,1].set_xlabel('Pred')
    ax[1,1].set_ylabel('True')
    lims = [np.min([ax[1,1].get_xlim(), ax[1,1].get_ylim()]),  # min of both axes
        np.max([ax[1,1].get_xlim(), ax[1,1].get_ylim()])]  # max of both axes
    ax[1,1].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    
    # # plot predictions and true for inds where time is way off
    # inds=np.where(np.abs(diffs[batchid,:,4])>43200)[0]
    # fig,ax=plt.subplots(2,3,figsize=(12,9))
    # ax[0,0].plot(regression[batchid,inds,0],y_test[0][batchid,inds,0],'o')
    # ax[0,0].set_title('Pred and True Lons')
    # ax[0,0].set_xlabel('Pred')
    # ax[0,0].set_ylabel('True')
    # lims = [np.min([ax[0,0].get_xlim(), ax[0,0].get_ylim()]),  # min of both axes
    #     np.max([ax[0,0].get_xlim(), ax[0,0].get_ylim()])]  # max of both axes
    # ax[0,0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    # #---------------
    # ax[0,1].plot(regression[batchid,inds,1],y_test[0][batchid,inds,1],'o')
    # ax[0,1].set_title('Pred vs True Lats')
    # ax[0,1].set_xlabel('Pred')
    # ax[0,1].set_ylabel('True')
    # lims = [np.min([ax[0,1].get_xlim(), ax[0,1].get_ylim()]),  # min of both axes
    #     np.max([ax[0,1].get_xlim(), ax[0,1].get_ylim()])]  # max of both axes
    # ax[0,1].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    # #---------------
    # ax[0,2].plot(regression[batchid,inds,2],y_test[0][batchid,inds,2],'o')
    # ax[0,2].set_title('Pred and True Depths')
    # ax[0,2].set_xlabel('Pred')
    # ax[0,2].set_ylabel('True')
    # lims = [np.min([ax[0,2].get_xlim(), ax[0,2].get_ylim()]),  # min of both axes
    #     np.max([ax[0,2].get_xlim(), ax[0,2].get_ylim()])]  # max of both axes
    # ax[0,2].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    # #---------------
    # ax[1,0].plot(regression[batchid,inds,3],y_test[0][batchid,inds,3],'o')
    # ax[1,0].set_title('Pred and True Mags')
    # ax[1,0].set_xlabel('Pred')
    # ax[1,0].set_ylabel('True')
    # lims = [np.min([ax[1,0].get_xlim(), ax[1,0].get_ylim()]),  # min of both axes
    #     np.max([ax[1,0].get_xlim(), ax[1,0].get_ylim()])]  # max of both axes
    # ax[1,0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    # #---------------
    # ax[1,1].plot(regression[batchid,inds,4],y_test[0][batchid,inds,4],'o')
    # ax[1,1].set_title('Pred and True Epoch')
    # ax[1,1].set_xlabel('Pred')
    # ax[1,1].set_ylabel('True')
    # lims = [np.min([ax[1,1].get_xlim(), ax[1,1].get_ylim()]),  # min of both axes
    #     np.max([ax[1,1].get_xlim(), ax[1,1].get_ylim()])]  # max of both axes
    # ax[1,1].plot(lims, lims, 'k-', alpha=0.75, zorder=0)