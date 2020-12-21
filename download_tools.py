#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:26:25 2020

cnn module

@author: amt
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from obspy.io.quakeml.core import Unpickler
from libcomcat.utils import get_phase_dataframe
from libcomcat.search import get_event_by_id
import pandas as pd
from datetime import timedelta
from obspy import Stream, read, UTCDateTime, Trace 
from obspy.clients.fdsn import Client
#from obspy.signal.cross_correlation import correlate_template
#from sklearn.cluster import DBSCAN

def make_training_data(df,sr,winsize,phase):   
    # make templates
    regional=df['Regional']
    if regional=='uw':
        client = Client("IRIS")
    elif regional=="nc":
        client = Client("NCEDC")    
    eventid = regional+str(df['ID'])
    detail = get_event_by_id(eventid, includesuperseded=True)
    phases = get_phase_dataframe(detail, catalog=regional)
    phases = phases[phases['Status'] == 'manual']
    if phase != 'N':
        phases = phases[phases['Phase'] == phase]
    # phases=phases[~phases.duplicated(keep='first',subset=['Channel','Phase'])]
    print(phases)
    st=Stream()
    for ii in range(len(phases)):
        tr=Stream()
        net=phases.iloc[ii]['Channel'].split('.')[0]
        sta=phases.iloc[ii]['Channel'].split('.')[1]
        comp=phases.iloc[ii]['Channel'].split('.')[2]
        pors=phases.iloc[ii]['Phase']
        #phase=phases.iloc[ii]['Phase']
        arr=UTCDateTime(phases.iloc[ii]['Arrival Time'])
        #print(int(np.round(arr.microsecond/(1/sr*10**6))*1/sr*10**6)==1000000)
        t1=arr-winsize/2
        t2=arr+winsize/2
        if phase =='N':
            t1-=120
            t2-=120
        try: # try to get the data
            tr = client.get_waveforms(net, sta, "*", comp, t1-1, t2+1)
            #print('Tr has length '+str(len(tr)))
        except:
            print("No data for "+net+" "+sta+" "+comp+" "+str(t1)+" "+str(t2))
        else:
            print("Data available for "+net+" "+sta+" "+comp+" "+str(t1)+" "+str(t2))
            try: # try to subsample the data
                tr.interpolate(sampling_rate=sr, starttime=t1)
            except:
                print("Data interp issues")
            else:
                tr.trim(starttime=t1, endtime=t2, nearest_sample=1, pad=1, fill_value=0)
        if len(tr) > 0:
            tr[0].stats.location=pors
            st+=tr 
    for tr in st: 
        # get rid of things that have lengths less than the desired length
        if len(tr.data) != sr*winsize+1:
            st.remove(tr)
    for tr in st: 
        # get rid of things that have all zeros
        if np.sum(tr.data)==len(tr.data):
            st.remove(tr)
    for tr in st: 
        # get rid of things that NaNs
        if np.sum(np.isnan(tr.data))>0:
            st.remove(tr)
    st.detrend()
    #plot_training_data_streams(st,sr)
    stout=np.zeros((len(st),sr*winsize+1))
    pors=np.zeros(len(st))
    for ii in range(len(st)):
        stout[ii,:]=st[ii].data
        if st[ii].stats.location=='P':
            pors[ii]=0
        if st[ii].stats.location=='S':
            pors[ii]=1
    return stout, pors

def make_training_data_3comp(df,sr,winsize,phase):   
    # make templates
    regional=df['Regional']
    if regional=='uw':
        client = Client("IRIS")
    elif regional=="nc":
        client = Client("NCEDC")    
    eventid = regional+str(df['ID'])
    detail = get_event_by_id(eventid, includesuperseded=True)
    phases = get_phase_dataframe(detail, catalog=regional)
    phases = phases[phases['Status'] == 'manual']
    if phase != 'N':
        phases = phases[phases['Phase'] == phase]
    # phases=phases[~phases.duplicated(keep='first',subset=['Channel','Phase'])]
    print(phases)
    stout=np.empty((0,3*(sr*winsize+1)))
    for ii in range(len(phases)):
        st=Stream()
        net=phases.iloc[ii]['Channel'].split('.')[0]
        sta=phases.iloc[ii]['Channel'].split('.')[1]
        comp=phases.iloc[ii]['Channel'].split('.')[2]
        comp=comp[:2]+'*'
        print(comp)
        #phase=phases.iloc[ii]['Phase']
        arr=UTCDateTime(phases.iloc[ii]['Arrival Time'])
        #print(int(np.round(arr.microsecond/(1/sr*10**6))*1/sr*10**6)==1000000)
        t1=arr-winsize/2
        t2=arr+winsize/2
        if phase =='N':
            t1-=120
            t2-=120
        try: # try to get the data
            st = client.get_waveforms(net, sta, "*", comp, t1-1, t2+1)
            #print('Tr has length '+str(len(tr)))
        except:
            print("No data for "+net+" "+sta+" "+comp+" "+str(t1)+" "+str(t2))
        else:
            print("Data available for "+net+" "+sta+" "+comp+" "+str(t1)+" "+str(t2))
            try: # try to subsample the data
                st.interpolate(sampling_rate=sr, starttime=t1)
            except:
                print("Data interp issues")
            else:
                st.trim(starttime=t1, endtime=t2, nearest_sample=1, pad=1, fill_value=0)        
        for tr in st: 
            # get rid of things that have lengths less than the desired length
            if len(tr.data) != sr*winsize+1:
                st.remove(tr)
        for tr in st: 
            # get rid of things that have all zeros
            if np.sum(tr.data)==len(tr.data):
                st.remove(tr)
        for tr in st: 
            # get rid of things that NaNs
            if np.sum(np.isnan(tr.data))>0:
                st.remove(tr)
        st.detrend()
        if len(st)==1:
            print(len(st[0].data))
            stout=np.append(stout,np.reshape(np.concatenate((st[0].data,st[0].data,st[0].data)),(1,(sr*winsize+1)*3)),axis=0)
        elif len(st)==3:
            print(len(st[0].data))
            print(len(st[1].data))
            print(len(st[2].data))
            stout=np.append(stout,np.reshape(np.concatenate((st[0].data,st[1].data,st[2].data)),(1,(sr*winsize+1)*3)),axis=0)
    return stout


def plot_training_data_streams(st,sr):
    plt.figure(figsize=(10,10))
    winlen=len(st[0].data)   
    templen=len(st[0].data)
    t=1/sr*np.arange(winlen)
    # plot template and detection relative to origin time
    stas=[]
    for ii in range(len(st)):
        clip=st[ii].data
        plt.plot(t,clip/np.max(1.5*np.abs(clip))+ii,color=(0.5,0.5,0.5))
        if st[ii].stats.location=='P':
            #print('its a P')
            plt.plot([t[-1]/2,t[-1]/2],[ii-0.5, ii+0.5],color=(0.5,0.0,0.0),linestyle='--')
        if st[ii].stats.location=='S':
            #print('its a S')
            plt.plot([t[-1]/2,t[-1]/2],[ii-0.5, ii+0.5],color=(0.0,0.0,0.5),linestyle='--')
        stas.append(st[ii].stats.station+"-"+st[ii].stats.channel)
    plt.xlim((0,t[-1]))
    plt.yticks(range(len(st)), stas)
    plt.xlabel('Time (s)')
    return None

def plot_training_data(st,sr,pors,wd):
    plt.figure(figsize=(10,10))
    winlen=wd*sr+1    
    templen=len(st[0].data)
    t=1/sr*np.arange(winlen)
    # plot template and detection relative to origin time
    stas=[]
    for ii in range(st.shape[0]):
        clip=st[ii,:]
        plt.plot(t,clip/np.max(1.5*np.abs(clip))+ii,color=(0.5,0.5,0.5))
        if pors[ii]==0:
            plt.plot([15,15],[ii-0.5, ii+0.5],color=(0.5,0.0,0.0),linestyle='--')
        if pors[ii]==1:
            plt.plot([15,15],[ii-0.5, ii+0.5],color=(0.0,0.0,0.5),linestyle='--')
    plt.xlim((0,t[-1]))
    plt.yticks(range(len(st)), stas)
    plt.xlabel('Time (s)')
    return None

def check_phase_info(df):
    exists=1
    regional=df['Regional']
    eventid=regional+str(df['ID'])
    try:
        detail=get_event_by_id(eventid, includesuperseded=True)
    except:
        exists=0
    else:
        try: 
            phases = get_phase_dataframe(detail, catalog=regional)
        except:
            exists=0    
    return exists

def simple(data):
    """
    Detrend signal simply by subtracting a line through the first and last
    point of the trace

    :param data: Data to detrend, type numpy.ndarray.
    :return: Detrended data. Returns the original array which has been
        modified in-place if possible but it might have to return a copy in
        case the dtype has to be changed.
    """
    # Convert data if it's not a floating point type.
    if not np.issubdtype(data.dtype, np.floating):
        data = np.require(data, dtype=np.float64)
    ndat = len(data)
    x1, x2 = data[0], data[-1]
    data -= x1 + np.arange(ndat) * (x2 - x1) / float(ndat - 1)
    return data