#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:42:34 2020

Earthquake phase pick generator

@author: amt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp2d
import datetime

def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km
    dlat = np.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) \
        * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = radius * c
    return d

testing=0
if testing:
    # # SO WE CAN REPRODUCE
    np.random.seed(42)
    # WANT SOME PLOTS
    plots=1
else:
    plots=0

# LOAD STATION INFORMATION
stas=pd.read_csv('station_master_with_elev.dat',delimiter=' ', usecols=(0,1,2,7,8,9), names=["Network","Station","Channel","Latitude","Longitude","Elevation"])
stas=stas.drop_duplicates(subset ="Station")
# Horizontals: BHE,BHN,EHE,EHN,HHE,HHN,EH1,EH2,BH3,HH3
# Verticals:BHZ,EHZ,HHZ
# BOTH:BH1,BH2,HH1,HH2

# LOAD VELOCITY MODELS
names=['c3', 'e3', 'j1', 'k3', 'n3', 'p4', 's4', 'O0', 'gil7'] 
for name in names:
    if name=='c3':
        dists, depths, parv, sarv = pickle.load( open( name+".pkl", "rb" ) )
    else:
        _,_,ptmp,stmp = pickle.load( open( name+".pkl", "rb" ) )
        parv=np.dstack((parv,ptmp))
        sarv=np.dstack((sarv,stmp))

# CREATE INTERP FUNCTIONS
c3p=interp2d(dists,depths,parv[:,:,0],fill_value=0)
c3s=interp2d(dists,depths,sarv[:,:,0],fill_value=0)
e3p=interp2d(dists,depths,parv[:,:,1],fill_value=0)
e3s=interp2d(dists,depths,sarv[:,:,1],fill_value=0)
j1p=interp2d(dists,depths,parv[:,:,2],fill_value=0)
j1s=interp2d(dists,depths,sarv[:,:,2],fill_value=0)
k3p=interp2d(dists,depths,parv[:,:,3],fill_value=0)
k3s=interp2d(dists,depths,sarv[:,:,3],fill_value=0)
n3p=interp2d(dists,depths,parv[:,:,4],fill_value=0)
n3s=interp2d(dists,depths,sarv[:,:,4],fill_value=0)
p4p=interp2d(dists,depths,parv[:,:,5],fill_value=0)
p4s=interp2d(dists,depths,sarv[:,:,5],fill_value=0)
s4p=interp2d(dists,depths,parv[:,:,6],fill_value=0)
s4s=interp2d(dists,depths,sarv[:,:,6],fill_value=0)
O0p=interp2d(dists,depths,parv[:,:,7],fill_value=0)
O0s=interp2d(dists,depths,sarv[:,:,7],fill_value=0)
gil7p=interp2d(dists,depths,parv[:,:,8],fill_value=0)
gil7s=interp2d(dists,depths,sarv[:,:,8],fill_value=0)

# SYNTHETIC SOURCE LOCATIONS
lat=np.linspace(39,51.5,1000000)# want uniform geographic sampling
latp=np.sin(lat*np.pi/180)
latp=latp/np.sum(latp)

# # do the shifts and make batches
def my_data_generator(lat,latp,c3p,c3s,e3p,e3s,j1p,j1s,k3p,k3s,n3p,n3s,p4p,p4s,s4p,s4s,O0p,O0s,gil7p,gil7s,batch_length=10000):
    while True:
        #defines coefficients for gmm
        a1,a2,a3,a4,a5=-1.96494392,  0.89260686, -0.12470324, -1.43533434, -0.00561138
        outfile=np.zeros((0,11))
        count=0
        ps=np.zeros((len(stas)))
        ss=np.zeros((len(stas)))
        jj=0
        sourceepoch=0
        while sourceepoch < 86400*7:
            #print(sourceepoch)
            sourcelon=np.random.uniform(-132.5,-116.5)
            sourcelat=np.random.choice(lat,p=latp)
            sourcedepth=np.random.uniform(0,100)
            sourcemag=np.random.uniform(1,6) # np.random.exponential(np.log(10)*1)
            dt=np.random.exponential(11564)
            if len(outfile)>0:
                sourceepoch=dt+sourceepoch
            else:
                sourceepoch=dt    
            # print("Trying earthquake # "+str(len(np.unique(outfile[:,9]))+1))
            # print(str(sourceepoch))
            dists=distance([sourcelat,sourcelon], [stas['Latitude'].values,stas['Longitude'].values])
            rhyp=np.sqrt(dists**2+sourcedepth**2) # dist from rupture in km
            gm=np.exp(a1 + a2*sourcemag + a3*(8.5-sourcemag)**2. + a4*np.log(rhyp) + a5*rhyp)
            # if plots and np.max(gm)>1e-06: # make sure recorded at at least one station
            #     fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(16,9))
            #     ax[0].plot(sourcelon,sourcelat,'ko',markersize=12)
            #     im0=ax[0].scatter(stas['Longitude'],stas['Latitude'],s=25,c=dists,marker='^')
            #     fig.colorbar(im0, ax=ax[0])
            #     ax[1].plot(sourcelon,sourcelat,'ko',markersize=12)
            #     im1=ax[1].scatter(stas['Longitude'],stas['Latitude'],s=25,c=rhyp,marker='^')
            #     fig.colorbar(im1, ax=ax[1])
            #     ax[2].plot(sourcelon,sourcelat,'ko',markersize=12)
            #     im2=ax[2].scatter(stas['Longitude'],stas['Latitude'],s=25,c=gm,marker='^')
            #     fig.colorbar(im2, ax=ax[2])
            if np.max(gm)>1e-06: # if ground motion is recordable
                for ii in range(len(stas)):       
                    if gm[ii]>=1e-06 and dists[ii]/300 < 0.5*np.random.uniform(): # if ground motion at station is recordable and adding a random drop term in here to account for missing/bad stations
                        dist=dists[ii]
                        if sourcelon<=-125:
                            mod='j1'
                            ps[ii]=j1p(dist,sourcedepth)
                            ss[ii]=j1s(dist,sourcedepth)
                        elif sourcelat<=42:
                            mod='gil7'
                            ps[ii]=gil7p(dist,sourcedepth)
                            ss[ii]=gil7s(dist,sourcedepth)
                        elif sourcelat>42 and sourcelat<=43:
                            mod='k3'
                            ps[ii]=k3p(dist,sourcedepth)
                            ss[ii]=k3s(dist,sourcedepth)
                        elif sourcelat>43 and sourcelat<=45.5:
                            mod='O0'
                            ps[ii]=O0p(dist,sourcedepth)
                            ss[ii]=O0s(dist,sourcedepth)
                        elif sourcelat>45.5 and sourcelat<=47 and sourcelon>-120.5:  
                            mod='e3'
                            ps[ii]=e3p(dist,sourcedepth)
                            ss[ii]=e3s(dist,sourcedepth)
                        elif sourcelat>47 and sourcelon>-120.5:  
                            mod='n3'
                            ps[ii]=n3p(dist,sourcedepth)
                            ss[ii]=n3s(dist,sourcedepth)
                        elif sourcelon>-122.69 and sourcelon<=-121.69 and sourcelat>45.69 and sourcelat<=46.69:
                            mod='s4'
                            ps[ii]=s4p(dist,sourcedepth)
                            ss[ii]=s4s(dist,sourcedepth)
                        elif sourcelon>-122.5:
                            mod='c3'
                            ps[ii]=c3p(dist,sourcedepth)
                            ss[ii]=c3s(dist,sourcedepth)
                        else:
                            mod='p4'
                            ps[ii]=p4p(dist,sourcedepth)
                            ss[ii]=p4s(dist,sourcedepth)
                        if ps[ii]>0:
                            # Pick Lon, Pick Lat, Pick Elev, P or S, Pick Time, Source Longitude, Source Latitude, Source Depth, Source Magnitude, Source Time])
                            evoutfile=np.zeros((2,11))
                            evoutfile[0,0]=stas.iloc[ii]['Longitude']
                            evoutfile[0,1]=stas.iloc[ii]['Latitude']
                            evoutfile[0,2]=stas.iloc[ii]['Elevation']
                            evoutfile[0,3]=1
                            evoutfile[0,4]=ps[ii]+sourceepoch #+np.random.normal(0, 0.05, 1000)
                            evoutfile[0,5]=sourcelon
                            evoutfile[0,6]=sourcelat
                            evoutfile[0,7]=sourcedepth
                            evoutfile[0,8]=sourcemag
                            evoutfile[0,9]=ps[ii]
                            evoutfile[0,10]=1
                            evoutfile[1,0]=stas.iloc[ii]['Longitude']
                            evoutfile[1,1]=stas.iloc[ii]['Latitude']
                            evoutfile[1,2]=stas.iloc[ii]['Elevation']
                            evoutfile[1,3]=0
                            evoutfile[1,4]=ss[ii]+sourceepoch #+np.random.normal(0, 0.05, 1000)
                            evoutfile[1,5]=sourcelon
                            evoutfile[1,6]=sourcelat
                            evoutfile[1,7]=sourcedepth
                            evoutfile[1,8]=sourcemag
                            evoutfile[1,9]=ss[ii]
                            evoutfile[1,10]=1
                            tmp=np.random.uniform() # make a random variable
                            if tmp < 0.1: # drop the P
                                outfile=np.append(outfile,evoutfile[1,:].reshape(1,-1),axis=0)
                            elif tmp >= 0.1 and tmp < 0.2: # drop the S
                                outfile=np.append(outfile,evoutfile[0,:].reshape(1,-1),axis=0)
                            elif tmp >= 0.2 and tmp < 0.3: # make the S a P
                                evoutfile[1,3]=1
                                outfile=np.append(outfile,evoutfile,axis=0)
                            elif tmp >= 0.3 and tmp < 0.4: # make the P a S
                                evoutfile[0,3]=0
                                outfile=np.append(outfile,evoutfile,axis=0)
                            else:
                                outfile=np.append(outfile,evoutfile,axis=0)
         
        # ADD SYNTHETIC NOISE
        fac=5
        outfilen=np.zeros((fac*len(outfile),11))
        count=0
        while count < fac*len(outfile):
            ii=np.random.choice(np.arange(len(stas)))
            outfilen[count,0]=stas.iloc[ii]['Longitude']
            outfilen[count,1]=stas.iloc[ii]['Latitude']
            outfilen[count,2]=stas.iloc[ii]['Elevation']
            outfilen[count,3]=np.random.choice([0,1])
            outfilen[count,4]=np.random.uniform(low=0,high=np.max(outfile[:,4]))
            outfilen[count,5]=0
            outfilen[count,6]=0
            outfilen[count,7]=0
            outfilen[count,8]=0
            outfilen[count,9]=0
            outfilen[count,10]=0
            count+=1
                 
        # COMBINE EQS AND NOISE
        allout=np.concatenate((outfile,outfilen))
        inds=np.argsort(allout[:,4])
        allout=allout[inds,:]  
        allout=allout[np.newaxis,:,:]
        #print(allout.shape)
        if allout.shape[1]>batch_length:
            allout=allout[:,:batch_length,:]
        #     print("clip")
        else:
            allout=np.append(allout,np.zeros((1,batch_length-allout.shape[1],11)),axis=1)
        #    print("add")
         
        if plots:
            # PLOT RESULTS
            pind=np.where(allout[0,:,3]==0)
            sind=np.where(allout[0,:,3]==1)
            # Pick Lon, Pick Lat, Pick Elev, P or S, Pick Time, Source Longitude, Source Latitude, Source Depth, Source Magnitude, Source Time, noise or sig])
            plt.figure()
            plt.scatter(allout[0,pind,4],allout[0,pind,1],s=25,c=allout[0,pind,-2],marker='+')
            plt.scatter(allout[0,sind,4],allout[0,sind,1],s=25,c=allout[0,sind,-2],marker='x')
            plt.plot(allout[0,np.where(allout[0,:,-1]!=0),-2],allout[0,np.where(allout[0,:,-1]!=0),-5],'ko')
            
            unisou=np.unique(outfile[:,9])
            bc=np.loadtxt('bc.outline')
            ca=np.loadtxt('ca.outline')
            ore=np.loadtxt('or.outline')
            wa=np.loadtxt('wa.outline')
            ida=np.loadtxt('id.outline')
            nv=np.loadtxt('nv.outline')
            for ind in range(len(unisou)):
                tmpsource=outfile[outfile[:,9]==np.unique(outfile[:,9])[ind]]
                plt.figure(figsize=(7,9))
                # Pick Lon, Pick Lat, Pick Elev, P or S, Pick Time, Source Longitude, Source Latitude, Source Depth, Source Magnitude, Source Time])
                plt.plot(tmpsource[0,5],tmpsource[0,6],'ko',markersize=5)
                plt.scatter(tmpsource[:,0],tmpsource[:,1],s=25,c=tmpsource[:,4],marker='^')
                plt.plot(wa[:,1],wa[:,0])
                plt.plot(bc[:,1],bc[:,0])
                plt.plot(ore[:,1],ore[:,0])
                plt.plot(ca[:,1],ca[:,0])
                plt.plot(ida[:,1],ida[:,0])
                plt.plot(nv[:,1],nv[:,0])
                plt.title('D'+str(np.round(10*tmpsource[0,7])/10)+'-M'+str(np.round(10*tmpsource[0,8])/10))
                plt.colorbar()

        yield(allout)                

# generate batch data
def get_generator():
    return my_data_generator(lat,latp,c3p,c3s,e3p,e3s,j1p,j1s,k3p,k3s,n3p,n3s,p4p,p4s,s4p,s4s,O0p,O0s,gil7p,gil7s,batch_length=10000)

if __name__=="__main__":
    my_data=get_generator()
    x=next(my_data)
