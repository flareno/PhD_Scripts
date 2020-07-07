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
###############################################################################

files = glob.glob(r'D:\F.LARENO.FACCINI\Preliminary Results\Behaviour\Group 8\15\Experiments\Training\LickFiles\*.lick')
names = [os.path.basename(x) for x in files]

print(names)    

for x in range(len(names)):
    names[x] = names[x].replace(".lick","")
    
############################## EXTRACTING RANDOM INTERVALS ####################

for file in range(len(names)):

    f = open(r'D:\F.LARENO.FACCINI\Preliminary Results\Behaviour\Group 8\15\Experiments\Training\LickFiles\%s.param' %(names[file]))
    lines = f.readlines()
    random = []
    r_time = []
    
    for line in range(len(lines)):
        if 'A2 to A3 transition duration:' in lines[line]:        
            random.append(lines[line])
        else:
            continue
    
    for i in random:
        temp__ = i.split(' ')[-1]
        temp__ = temp__.split('\n')[0]
        r_time.append(temp__)
     
    r_time = np.asarray(r_time)
     
############################# SPEED OF THE WHEEL ##############################
#
##for file in range(len(names)):
#   
#    file_path=(r"D:\F.LARENO.FACCINI\Preliminary Results\DATA\Group 8\15\Experiments\Training\LickFiles\%s.coder" %(names[file]))  
#
#    A = (pd.read_csv(file_path, sep="\t"))
#    A = np.array(A)
#    
#    B = [[A[i][0], float(A[i][1].replace(',','.'))] for i in range(len(A))]
#    B = np.array(B)
#    
#    K = np.empty((len(r_time), 250))
#    K[:] = np.nan
#    
#    N_TRIALS = len(r_time)    
#
    fig, ax = plt.subplots(3,1,figsize=(10,8), sharex=True)
#
#    for h in range(N_TRIALS):
#        T= [B[:,1][j] for j in range(len(B)) if B[:,0][j]==h]
#    
#        if len(T)>0:
#            ax[0].scatter(T, np.linspace(-h,-h, len(T)), color='black', s=2,alpha=0.1)
#            for k in range(len(T)):
#                K[h,k] = T[k]
#    
#    K = K[np.logical_not(np.isnan(K))]  # Removes NaN Values 
    
############################### LICK SCATTER ##################################       
            
    path=(r'D:\F.LARENO.FACCINI\Preliminary Results\Behaviour\Group 8\15\Experiments\Training\LickFiles\%s.lick' %(names[file]))
    
    C = (pd.read_csv(path, sep="\t", header=None))
    C = np.array(C)
    
    D = [[C[i][0], float(C[i][1].replace(',','.'))] for i in range(len(C))]
    D = np.array(D)
        
    Y = []
    
    N_TRIALS = len(r_time)
    
    
    for p in range(N_TRIALS):
        temp__ = [D[:,1][j] for j in range(len(D)) if D[:,0][j]==p]
    
        if len(temp__)>0:
#            ax[0].scatter(temp__, np.linspace(-p,-p, len(temp__)), color='red', s=2,alpha=0.4)
                     
            Y.append(np.array(temp__))
                
        else:
            Y.append(np.nan)
   
     
    #R = [Y[i] for i in r_time]
    
    
#    shape = ((len(r_time)), 2)
#    R = np.empty(shape)
#    
#    for x in range(len(r_time)):
#        #np.append(R[x][0],Y[x])
#        np.append(R[x,1],r_time[x])
    R = []

    for p in range(len(r_time)):
        temp__ = [Y[:,1][j] for j in range(len(r_time)) if Y[:,0][j]==p]
    
        if len(temp__)>0:
            ax[0].scatter(temp__, np.linspace(-p,-p, len(temp__)), color='red', s=2,alpha=0.4)
                     
            R.append(np.array(temp__))
                
        else:
            R.append(np.nan)
    #np.stack((r_time, Y), axis=-1)
    
################################### CUMULATIVE #################################
#    
#    cm = ['black', 'red', 'blue', 'seagreen', 'gold', 'purple', 'deepskyblue', 'lightsalmon', 'hotpink']
#    
#    n_bins = 800                                                               # To have a bining of 10 ms!!
#    
#    n_trials = len(r_time)
#    step = 10
#    bins = np.arange(0, n_trials+2*step, step)
#    
#
#    for i in range(len(bins)):
#        
#        if i == len(bins)-1:
#            break
#        
#        else:    
#            start = bins[i]
#            stop = bins[i+1]
#            
#            binned_array = [] 
#            to_iterate = np.arange(start,stop+1,1)                             # Crea un array con i 50 indici
#            
#            for j in range(len(to_iterate)):
#                
#                if np.isnan(Y[to_iterate[j]]).any() == True:                   # ignora i trial con NaN
#                    continue
#                else:                        
#                    binned_array.append(np.asarray(Y[to_iterate[j]]))
#            
#            if len(binned_array)>1:         
#                binned_array = np.concatenate(binned_array,axis = 0)               # ho un array unico con tutti i valori del gruppo di 50 trials
#            
#            
#                binned_array = np.sort(binned_array)                               # ordino i valori in ordine crescente
#                       
#                ax[1].hist(binned_array, n_bins, histtype='step', cumulative=True, density=True, label=' %s - %s'%(start,stop),linewidth=1.2)
# 
#   
########################### CUMULATIVE OF THE WHEEL ############################
#
#    for l in range(len(bins)):
#        
#        if l == len(bins)-1:
#            break
#        
#        else:    
#            start = bins[l]
#            stop = bins[l+1]
#            
#            array = [] 
#            iterate = np.arange(start,stop+1,1)                             # Crea un array con i 50 indici
#            
#            for j in range(len(iterate)):
#                
#                if np.isnan(K[iterate[j]]).any() == True:                   # ignora i trial con NaN
#                    continue
#                else:                        
#                    array.append(K[iterate[j]])
#                    
#            #array = np.concatenate(array,axis = 0)               # ho un array unico con tutti i valori del gruppo di 50 trials
#            array = np.sort(array)                               # ordino i valori in ordine crescente
#                       
#            ax[2].hist(array, n_bins, histtype='step', cumulative=True, density=True,linewidth=1.2,alpha=0.8)
# 
    
############################ PLOTTING DETAILS #################################
    
    
    title=path.split('LickFiles\\')[1]
    ax[0].set_title(title)
           
    major_xticks = np.arange(0, 8, 1)
    minor_xticks = np.arange(0, 8, 0.1)
    major_yticks = np.arange(0, 1.1, 0.2)
    minor_yticks = np.arange(0, 1.2, 0.03)
    
    
    ax[0].set_ylabel("Trial number")
    ax[0].grid()
    ax[0].plot((2.68,2.68),(0,-N_TRIALS),color="red")
    ax[0].plot((1.5,1.5),(0,-N_TRIALS),color="blue")
    ax[0].plot((2,2),(0,-N_TRIALS),color="yellow")
    ax[0].plot((0.5,0.5),(0,-N_TRIALS),color="green")
    ax[0].plot((2.88,2.88),(0,-N_TRIALS),color="purple")
    
    
    ax[1].set_xticks(major_xticks)
    ax[1].set_xticks(minor_xticks, minor=True)
    ax[1].set_yticks(major_yticks)
    ax[1].set_yticks(minor_yticks, minor=True)
    ax[1].legend(loc='upper left', prop={'size':7})
    ax[1].grid(which='minor', alpha=0.6)       
    ax[1].plot((2.68,2.68),(0,1),color="red", alpha=0.6)
    ax[1].plot((2.88,2.88),(0,1),color="purple", alpha=0.6)
    ax[1].set_ylabel("P of having a lick event")              
    
    
    ax[2].set_xticks(major_xticks)
    ax[2].set_xticks(minor_xticks, minor=True)
    ax[2].set_yticks(major_yticks)
    ax[2].set_yticks(minor_yticks, minor=True)
    ax[2].grid(which='minor', alpha=0.6)       
    ax[2].plot((2.68,2.68),(0,1),color="red", alpha=0.6)
    ax[2].plot((2.88,2.88),(0,1),color="purple", alpha=0.6)
    ax[2].set_xlabel("Time of the Trial (s)")
    ax[2].set_ylabel("Wheel")              


   # plt.savefig(r"D:\F.LARENO.FACCINI\Preliminary Results\DATA\Group 3\%s_Sliced_Cumulative_Wheel.pdf" %(names[file]))                  