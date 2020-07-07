# -*- coding: utf-8 -*-
import numpy as np 
from numpy import genfromtxt as gen
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import glob
import os
matplotlib.rcParams['pdf.fonttype'] = 42  

"""
Created on Mon Apr 16 18:32:17 2018

@author: F.LARENO-FACCINI
"""

############################# SPEED OF THE WHEEL ##############################

files = glob.glob(r"D:\F.LARENO.FACCINI\Preliminary Results\DATA\Group 8\16\Experiments\Lick Files\*.coder")
names = [os.path.basename(x) for x in files]

for x in range(len(names)):
    names[x] = names[x].replace(".coder","")


for file in range(len(names)):
    file_path=(r"%s" %(files[file]))
    file_path=(r"D:\F.LARENO.FACCINI\Preliminary Results\DATA\Group 8\16\Experiments\Lick Files\%s.coder" %(names[file]))
#    
#    A = (pd.read_csv(file_path, sep="\t"))
#    A = np.array(A)
#    
#    B = [[A[i][0], float(A[i][1].replace(',','.'))] for i in range(len(A))]
#    B = np.array(B)
#
#    K = np.empty((451, 250))
#    K[:] = np.nan
#    
#    N_TRIALS = 451
#    
    fig, ax = plt.subplots(2,1,figsize=(10,8))
#    
#    for i in range(N_TRIALS):
#        T= [B[:,1][j] for j in range(len(B)) if B[:,0][j]==i]
#    
#        if len(T)>0:
#            ax[0].scatter(T, np.linspace(-i,-i, len(T)), color='black', s=2,alpha=0.5)
#            for k in range(len(T)):
#                K[i,k] = T[k]
#
#    K = K[np.logical_not(np.isnan(K))]  # Removes NaN Values 
#            
            
################################## LICK #######################################    

    path=(r"D:\F.LARENO.FACCINI\Preliminary Results\DATA\Group 8\16\Experiments\Lick Files\%s.lick" %(names[file]))   
         
    C = (pd.read_csv(path, sep="\t",header=None))
    C = np.array(C)
    
    D = [[C[i][0], float(C[i][1].replace(',','.'))] for i in range(len(C))]
    D = np.array(D)
       
    Y = np.empty((451, 250))
    Y[:] = np.nan
    
    N_TRIALS = 451
    
    for i in range(N_TRIALS):
        temp__ = [D[:,1][j] for j in range(len(D)) if D[:,0][j]==i]
    
        if len(temp__)>0:
            ax[0].scatter(temp__, np.linspace(-i,-i, len(temp__)), color='red', s=2,alpha=0.3)
            for y in range(len(temp__)):
                Y[i,y] = temp__[y]
            
    Y = Y[np.logical_not(np.isnan(Y))]  # Removes NaN Values 
            
    title=file_path.split('Lick Files\\')[1]
    ax[0].set_title(title)
    ax[0].set_xlabel("time from begining or trial (s)")
    ax[0].set_ylabel("Trial number")
    ax[0].grid()
    ax[0].plot((2.68,2.68),(0,-N_TRIALS),color="red")
    ax[0].plot((1.5,1.5),(0,-N_TRIALS),color="blue")
    ax[0].plot((2,2),(0,-N_TRIALS),color="yellow")
    ax[0].plot((0.5,0.5),(0,-N_TRIALS),color="green")
    ax[0].plot((2.88,2.88),(0,-N_TRIALS),color="purple")
    
   
##################################### PSTH ####################################

    n_bins = 300
    #n,bins,patches = ax[1].hist(K,n_bins, normed=1, histtype='step', cumulative=True,label='Cumulative')
    #ax[1].hist(K, n_bins, histtype='bar', normed=False, color='black', alpha=0.4,label='Wheel')
    ax[1].hist(Y, n_bins, histtype='step', normed=True, color='red', alpha=0.8, cumulative=True)
    #ax[1].legend(loc='best')
    
    ax[1].plot((2.68,2.68),(0,1),color="red", alpha=0.3)
    #ax[1].plot((1.5,1.5),(0,1),color="blue")
    #ax[1].plot((2,2),(0,1),color="yellow")
    #ax[1].plot((0.5,0.5),(0,1),color="green")
    ax[1].plot((2.88,2.88),(0,1),color="purple", alpha=0.3)
    
    
    
    #plt.savefig(r"D:\F.LARENO.FACCINI\Preliminary Results\DATA\June Group\6367\Experiments\Lick Files\%s_sum_PSTH.pdf" %(names[file]))
