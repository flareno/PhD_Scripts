# -*- coding: utf-8 -*-
import numpy as np 
from numpy import genfromtxt as gen
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.rcParams['pdf.fonttype'] = 42  

"""
Created on Tue Mar 19 16:56:24 2019

@author: F.LARENO-FACCINI
"""

file_path=(r"D:\F.LARENO.FACCINI\Preliminary Results\DATA\Group 7\6352\Experiments\Lick Files\6352_2018_06_01_11_03_35.lick")

A = (pd.read_csv(file_path, sep="\t",header=None))
A = np.array(A)
B = [[A[i][0], float(A[i][1].replace(',','.'))] for i in range(len(A))]
B = np.array(B)

Y = []
ant = []

N_TRIALS = 510
step = 16
bins = np.arange(0, N_TRIALS+step, step)
binning = np.arange(0,5.01,0.01)

for i in range(len(bins)):
        
        if i == len(bins)-1:           
            break
        
        else:    
            start = bins[i]
            stop = bins[i+1]
            to_iterate = np.arange(start+1,stop+1,1)
            #print(i)
            
            fig, ax = plt.subplots(4,4, figsize=(15, 6), facecolor='w', edgecolor='k', sharex=True, sharey=True)
            fig.add_subplot(111, frameon=False)            
            ax = ax.ravel()
            
            for p in range(len(to_iterate)):
                
                if to_iterate[p] < 71:
                    continue
                
                else:
                    temp__ = [B[:,1][j] for j in range(len(B)) if B[:,0][j]==(to_iterate[p])]
                    print(to_iterate[p])
                    n, binns, patches = ax[p].hist(temp__, bins=binning, rwidth=0.3)
                    ax[p].vlines(x=(2.68,2.68), ymin=0, ymax=3,color="red", alpha=0.2)
                    ax[p].vlines(x=(2.88,2.88), ymin=0, ymax=3,color="purple", alpha=0.2)
                    ax[p].title.set_text('Trial %s' %(to_iterate[p]))
    
                    if len(temp__)>0:
                        Y.append(np.array(temp__))
                    else:
                        Y.append(np.nan)
                   
                    ant.append(np.array(n))
                    #print(Y)
            
            
            plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            plt.grid(False)
            plt.ylabel('Number Count')
            plt.xlabel('Time (s)')
            plt.tight_layout()        
            #plt.savefig(r"D:\F.LARENO.FACCINI\Preliminary Results\DATA\Group 7\6352\Selected trials\%s-%s.pdf" %(start,stop))
            #plt.close()

plt.figure()
tot = np.sum(ant,axis=0)
plt.xlabel('Time (s)')
plt.ylabel('Number Count')
plt.bar(np.arange(0,5,0.01),tot, width=0.009)
plt.vlines(x=(2.68,2.68), ymin=0, ymax=(np.amax(tot)),color="red", alpha=0.8, linewidth=0.4)
plt.vlines(x=(2.88,2.88), ymin=0, ymax=(np.amax(tot)),color="purple", alpha=0.8, linewidth=0.4)
#plt.savefig(r"D:\F.LARENO.FACCINI\Preliminary Results\DATA\Group 7\6352\Selected trials\PSTH_tot.pdf")

plt.figure()
plt.imshow(ant, cmap='jet', vmin=0, vmax=(np.amax(ant)/2))
plt.colorbar()
plt.xlabel('Time (cs)')
plt.ylabel('Trial Number')
plt.vlines(x=(268,268), ymin=0, ymax=450,color="red", alpha=0.8, linewidth=0.4)
plt.vlines(x=(288,288), ymin=0, ymax=450,color="purple", alpha=0.8, linewidth=0.4)
#plt.savefig(r"D:\F.LARENO.FACCINI\Preliminary Results\DATA\Group 7\6352\Selected trials\density_flow.pdf")
