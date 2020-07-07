# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:56:24 2019

@author: F.LARENO-FACCINI


Separate behavioural trials between OPTOGENETIC and CONTROL trials.
Concatenate all the trials of different files (maintaining the division between opto and control).
Plot the raster and PSTH of the whole concatenated trials.
"""
from __future__ import division, print_function

import matplotlib.pyplot as plt
import extrapy.Behaviour as bv
import glob
import os
import numpy as np

import quantities as pq
from quantities import Hz, s, ms
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.stats
import scipy.signal
import neo
from neo.core import SpikeTrain
import elephant.spike_train_generation as Spike
import elephant.statistics as stat
import elephant.conversion as conv
import elephant.kernels as kernels
import warnings
from mpl_toolkits.mplot3d import Axes3D
import bisect as bi
from scipy.ndimage.filters import gaussian_filter1d


protocol = 'P18'
mouse = '6409'
nb_control_trials = 30
nb_stim_trials = 30
op_t = 0.03
len_reward = 0.150
len_aspiration = 0.7
time_trial = 10
stim_len = 1

files = glob.glob(fr"D:/F.LARENO.FACCINI/Preliminary Results/Behaviour/Group 14/{mouse}/Fixed Delay/{protocol}/*.lick")
names = [os.path.basename(x) for x in files]

for x in range(len(names)):
    names[x] = names[x].replace(".lick","")

Concat_trials_nostim = [0]  
Concat_licks_nostim = [0]
Concat_trials_stim = [0]  
Concat_licks_stim = [0]

for giri,file in enumerate(names, start=1):    
    path = f"D:/F.LARENO.FACCINI/Preliminary Results/Behaviour/Group 14/{mouse}/Fixed Delay/{protocol}/{file}.lick"
    lick = bv.load_lickfile(path)
  
    # ===========================================================================
    # ======================   SEPARATE OPTO AND CONTROL   ======================
    # ===========================================================================
    nostim = []
    stim = []
    
    [nostim.append([t,l]) for t,l in lick if t <= nb_control_trials]
    nostim = np.asarray(nostim)
    [stim.append([t,l]) for t,l in lick if t > nb_control_trials]
    stim = np.asarray(stim)
 
    # print(len(nostim), len(stim))
    
    # ===========================================================================
    # ==========================   CONCATENATE   ================================ 
    # ===========================================================================
    # No Stim
    if len(nostim) > 0:
        nostim[:,0] = [t+Concat_trials_nostim[-1] for t,l in nostim]
        if nostim[-1,0] != nb_control_trials*giri:
            idx = np.int(nb_control_trials*giri-nostim[-1,0])
            for x in range(idx):
                addition = np.array([(nostim[-1,0]+1),np.nan])
                nostim = np.vstack((nostim,addition))
    else:
        nostim = np.array([nb_control_trials*giri,np.nan]).reshape((1,2))
        
    Concat_trials_nostim = np.append(Concat_trials_nostim,nostim[:,0])
    Concat_licks_nostim = np.append(Concat_licks_nostim,nostim[:,1])

    # Stim
    # stim[:,0] = [t+Concat_trials_stim[-1] for t,l in stim]
    for abc, (t,l) in enumerate(stim):
        # stim[abc,0] = np.int(nb_control_trials*giri)

        if (giri-1) == 0:
            stim[abc,0] = t-nb_control_trials
        else:
            delta = t-(nb_control_trials+1)
            stim[abc,0] = nb_stim_trials*(giri-1)+delta+1
    if len(stim)>0:
        if stim[-1,0] != nb_stim_trials*giri:
            iddx = np.int(nb_stim_trials*giri-stim[-1,0])
            for x in range(iddx):
                add = np.array([(stim[-1,0]+1),np.nan])
                stim = np.vstack((stim,add))
    else:
        stim = np.array([nb_stim_trials*giri,np.nan]).reshape((1,2))

    Concat_trials_stim = np.asarray(np.append(Concat_trials_stim,stim[:,0]))
    Concat_licks_stim = np.asarray(np.append(Concat_licks_stim,stim[:,1]))

# No Stim
Concat_trials_nostim = Concat_trials_nostim.astype(int)
Concat_trials_nostim = np.delete(Concat_trials_nostim, 0)
Concat_licks_nostim = np.delete(Concat_licks_nostim, 0)
nostim_concat = []
[nostim_concat.append([Concat_trials_nostim[i], Concat_licks_nostim[i]]) for i in range(len(Concat_trials_nostim))]
nostim_concat = np.asarray(nostim_concat)

ns_dove = np.argwhere(np.isnan(nostim_concat))
ns_dove = [ns_dove[i][0] for i in range(len(ns_dove))]

new_nostim_concat = []
for idx, (t,l) in enumerate(nostim_concat):
   if idx in ns_dove:
       pass
   else:
       new_nostim_concat.append((t,l))
new_nostim_concat = np.asarray(new_nostim_concat)

# Stim
Concat_trials_stim = Concat_trials_stim.astype(int)
Concat_trials_stim = np.delete(Concat_trials_stim, 0)
Concat_licks_stim = np.delete(Concat_licks_stim, 0)
stim_concat = []
[stim_concat.append([Concat_trials_stim[i], Concat_licks_stim[i]]) for i in range(len(Concat_trials_stim))]
stim_concat = np.asarray(stim_concat)

s_dove = np.argwhere(np.isnan(stim_concat))
s_dove = [s_dove[i][0] for i in range(len(s_dove))]
new_stim_concat = []
for idx, (t,l) in enumerate(stim_concat):
   if idx in s_dove:
       pass
   else:
       new_stim_concat.append((t,l))
new_stim_concat = np.asarray(new_stim_concat)


############################################################################
###################   PLOTTING DETAILS OF CONTROL   ########################
############################################################################
fig, ax = plt.subplots(3,2, figsize=(15,8),sharex=True, sharey='row')

bv.scatter_lick(new_nostim_concat, ax=ax[0,0])
ns_n,ns_bins,ns_patches = bv.PSTH_lick(new_nostim_concat, lentrial = time_trial, color='r', ax=ax[1,0])
d=0.5
ax[0,0].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
ax[0,0].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
ax[0,0].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
ax[0,0].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
ax[1,0].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
ax[1,0].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
ax[1,0].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
ax[1,0].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
# ax[1].axvline(2.07+d, linestyle='--', color='k', linewidth=0.7, alpha=0.4)
ax[1,0].set_ylabel('Density')
ax[0,0].set_title('Control Trials of {}_{}'.format(mouse,protocol))

# PLOTTING THE ENVELOPE (which is n, the y value of each point of the PSTH)
ns_binning = []
for h in range(len(ns_bins)):
    if h>0:
        ns_binning.append(np.mean((ns_bins[h],ns_bins[h-1])))
ns_step = time_trial/len(ns_binning)
ns_x = np.arange(0,time_trial,ns_step)

# smoothening the envelope
ns_nsmoothed = gaussian_filter1d(ns_n, sigma=3.5)
ax[2,0].plot(ns_x, ns_nsmoothed, color='k',alpha=0.9,linewidth=0.5)

ax[2,0].set_title('Envelope')
ax[2,0].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
ax[2,0].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
ax[2,0].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
ax[2,0].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
ax[2,0].set_ylabel('Density')
ax[2,0].set_xlabel('Time (s)')



# ===========================================================================
# ====================   PLOTTING DETAILS OF OPTO   =========================
# ===========================================================================
# fig2, ax2 = plt.subplots(3, sharex=True)
bv.scatter_lick(new_stim_concat, ax=ax[0,1])
s_n, s_bins,s_patches = bv.PSTH_lick(new_stim_concat, lentrial = time_trial, color='r', ax=ax[1,1])
d=0.5

ax[0,1].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
ax[0,1].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
ax[0,1].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
ax[0,1].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
ax[1,1].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
ax[1,1].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
ax[1,1].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
ax[1,1].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
# ax2[1].axvline(2.07+d, linestyle='--', color='k', linewidth=0.7, alpha=0.4)
ax[1,1].set_ylabel('Density')
ax[0,1].set_title('Photostimulated Trials of {}_{}'.format(mouse,protocol))

# PLOTTING THE ENVELOPE (which is n, the y value of each point of the PSTH)
s_binning = []
for k in range(len(s_bins)):
    if k>0:
        s_binning.append(np.mean((s_bins[k],ns_bins[k-1])))
s_step = time_trial/len(s_binning)
s_x = np.arange(0,time_trial,s_step)

# smoothening the envelope
s_nsmoothed = gaussian_filter1d(s_n, sigma=3.5)
ax[2,1].plot(s_x, s_nsmoothed, color='k',alpha=0.9,linewidth=0.5)

ax[2,1].set_title('Envelope')
ax[2,1].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
ax[2,1].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
ax[2,1].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
ax[2,1].axvspan(2+len_reward+op_t+d+0.05,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
ax[2,1].set_ylabel('Density')
ax[2,1].set_xlabel('Time (s)')

ax[0,1].axvspan(1.5, 1.5+stim_len,color='skyblue', alpha = 0.4) # Stim
ax[1,1].axvspan(1.5, 1.5+stim_len,color='skyblue', alpha = 0.4) # Stim
ax[2,1].axvspan(1.5, 1.5+stim_len,color='skyblue', alpha = 0.4) # Stim

ax[2,0].axvline(0.5+op_t+d, linestyle='--', color='r', alpha=0.2)
ax[2,1].axvline(0.5+op_t+d, linestyle='--', color='r', alpha=0.2)


fig.savefig(fr"D:/F.LARENO.FACCINI/Preliminary Results/Behaviour/Group 14/Plots/PSTH_{mouse}_{protocol}_Stim_vs_NoStim.pdf")
plt.close('all')

'''
####################################################################################################
####################################################################################################
#                                       SUCCESS AND ANTICIPATION
####################################################################################################
####################################################################################################

# =============================================================================
# Some variables
# =============================================================================
delay = 0.5         #  include delay + 
OT = 0.05           #  valve opening time (OT in Labview)
binsize=0.02*s       #  for the PSTH
sampling=0.05*s       #  sampling for the convolution
sampling2 = 0.05*s    #  sampling for the kernel. Probably leave as sampling
t_stop = 10*s      #  file duration: extend beyond the real duration 
debut_win = int(2/sampling+OT/sampling)       # window for the detection of the success lick
fin_win = int(3.5/sampling+OT/sampling)+int(0.4/sampling) 
n_sd = 3            # number of SD for the selection above the baseline 

# =============================================================================
# prepare for spiketrain convolution
#  1-calculate the number of lick per trial 
# =============================================================================
#NO STIM
nostim_number_value_per_trial = np.bincount(new_nostim_concat[:,0].astype(int))
nostim_number_value_per_trial = np.delete(nostim_number_value_per_trial, 0)
nostim_trials=np.arange(len(nostim_number_value_per_trial))
#STIM
stim_number_value_per_trial = np.bincount(new_stim_concat[:,0].astype(int))
stim_number_value_per_trial = np.delete(stim_number_value_per_trial, 0)
stim_trials=np.arange(len(stim_number_value_per_trial))

# =============================================================================
#   2-prepare matrix for all convolved trials 
# Z is the matrix row:trials, columns:nulber of points of the the convoluted trial
# =============================================================================                                                        
# NO STIM
nsy =[[i*sampling] for i in range(int(t_stop/sampling))]    # genera un array con il tempo di ogni punto (il loop dá il numero di punti tot e poi con (i*samp_rate) ottiene il tempo di ogni punto)                          
nsX,nsY = np.meshgrid(nsy,nostim_trials)
nsZ = np.zeros((len(nostim_trials),len(nsy)))
# STIM
sy =[[i*sampling] for i in range(int(t_stop/sampling))]    # genera un array con il tempo di ogni punto (il loop dá il numero di punti tot e poi con (i*samp_rate) ottiene il tempo di ogni punto)                          
sX,sY = np.meshgrid(sy,stim_trials)
sZ = np.zeros((len(stim_trials),len(sy)))

# =============================================================================
# loop and convolve all trials
# ============================================================================
nostim_debut_train = 0
# NO STIM
for i in range(len(nostim_number_value_per_trial)):
    train = new_nostim_concat[nostim_debut_train:nostim_debut_train+nostim_number_value_per_trial[i],1]*s         
    SpikeT = SpikeTrain(train,t_start=0.0*s,t_stop=t_stop)                   
    kernel = kernels.GaussianKernel(sigma = sampling2, invert=False)
    rate = stat.instantaneous_rate(SpikeT, sampling, kernel, cutoff=5.0, t_start=None, t_stop=None, trim=False)  # Negative inst. rate means a reduction of firing rate in that bin
    nostim_debut_train = nostim_debut_train + nostim_number_value_per_trial[i]
    for j in range(len(rate)): 
        nsZ[i, j] = rate[j]
#STIM
stim_debut_train = 0
for i in range(len(stim_number_value_per_trial)):
    train = new_stim_concat[stim_debut_train:stim_debut_train+stim_number_value_per_trial[i],1]*s       # all licks of one trial  
    SpikeT = SpikeTrain(train,t_start=0.0*s,t_stop=t_stop)                   
    kernel = kernels.GaussianKernel(sigma = sampling2, invert=False)
    rate = stat.instantaneous_rate(SpikeT, sampling, kernel, cutoff=5.0, t_start=None, t_stop=None, trim=False)  # Negative inst. rate means a reduction of firing rate in that bin
    stim_debut_train = stim_debut_train + stim_number_value_per_trial[i]
    for j in range(len(rate)): 
        sZ[i, j] = rate[j]

# =============================================================================
# Exclude/Select trials: null trial + failures
# =============================================================================
# NO STIM
nostim_success_trial, nostim_fail_trial = [], []
for i in range(len(nostim_number_value_per_trial)):
    if (np.sum(nsZ[i,:])<1 or np.mean(nsZ[i,debut_win:fin_win])<np.mean(nsZ[i,0:debut_win-1])+n_sd*np.std(nsZ[i,0:debut_win-1])): #ATTENTION DEPEND ON LEN(RATE) WHICH DEPEND ON SAMPLING 
        nostim_fail_trial.append(i+1)
        for j in range(len(rate)): 
            nsZ[i, j] = float('nan')
    else:
        nostim_success_trial.append(i+1)
        
nsZ = nsZ[~np.isnan(nsZ).any(axis=1)]
nsx = np.arange(np.size(nsZ,0))
nsX,nsY = np.meshgrid(nsy,nostim_trials)


# STIM
stim_success_trial, stim_fail_trial = [], []
for i in range(len(stim_number_value_per_trial)):
    if (np.sum(sZ[i,:])<1 or np.mean(sZ[i,debut_win:fin_win])<np.mean(sZ[i,0:debut_win-1])+n_sd*np.std(sZ[i,0:debut_win-1])): #ATTENTION DEPEND ON LEN(RATE) WHICH DEPEND ON SAMPLING 
        stim_fail_trial.append(i+1)
        for j in range(len(rate)): 
            sZ[i, j] = float('nan')
    else:
        stim_success_trial.append(i+1)

sZ = sZ[~np.isnan(sZ).any(axis=1)]
sx = np.arange(np.size(sZ,0))
sX,sY = np.meshgrid(sy,stim_trials)


# ==========================================================================================
# ENVELOPE OF SUCCESSES AND FAILS
# ==========================================================================================
###########
# NO STIM #
###########
# nostim_success, nostim_fail = [], []
# for t,l in new_nostim_concat:
#     if t in nostim_success_trial:
#         nostim_success.append((t,l))
#     else:
#         nostim_fail.append((t,l))

# nostim_fail = np.asarray(nostim_fail)
# nostim_success = np.asarray(nostim_success)

# fig4, ax4 = plt.subplots(3,2, figsize=(15,8), sharex=True)

# ## Plot Success
# if len(nostim_success) >0:
#     bv.scatter_lick(nostim_success, ax=ax4[0,0])
#     nss_n,nss_bins,nss_patches = bv.PSTH_lick(nostim_success, lentrial=time_trial, color='r', ax=ax4[1,0])
#     d=0.5
#     ax4[0,0].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
#     ax4[0,0].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
#     ax4[0,0].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
#     ax4[0,0].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
#     ax4[1,0].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
#     ax4[1,0].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
#     ax4[1,0].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
#     ax4[1,0].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
#     ax4[1,0].set_ylabel('Number count')
#     ax4[0,0].set_title('Success'.format(mouse,protocol,power))
#     # Envelope Success (which is n, the y value of each point of the PSTH)
#     nss_binning = []
#     for h in range(len(nss_bins)):
#         if h>0:
#             nss_binning.append(np.mean((nss_bins[h],nss_bins[h-1])))
#     nss_step = time_trial/len(nss_binning)
#     nss_x = np.arange(0,time_trial,nss_step)
#     nss_nsmoothed = gaussian_filter1d(nss_n, sigma=3.5)
#     ax4[2,0].plot(nss_x, nss_nsmoothed, color='k',alpha=0.9,linewidth=0.5)
#     ax4[2,0].set_title('Envelope')
#     ax4[2,0].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
#     ax4[2,0].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
#     ax4[2,0].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
#     ax4[2,0].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
#     ax4[2,0].set_ylabel('Number count')
#     ax4[2,0].set_xlabel('Time (s)')
    
# ## Plot Fail
# if len(nostim_fail) >0:
#     bv.scatter_lick(nostim_fail, ax=ax4[0,1])
#     nsf_n,nsf_bins,nsf_patches = bv.PSTH_lick(nostim_fail, lentrial = time_trial, color='r', ax=ax4[1,1])
#     d=0.5
#     ax4[0,1].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
#     ax4[0,1].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
#     ax4[0,1].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
#     ax4[0,1].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
#     ax4[1,1].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
#     ax4[1,1].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
#     ax4[1,1].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
#     ax4[1,1].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
#     ax4[1,1].set_ylabel('Number count')
#     ax4[0,1].set_title('Fail'.format(mouse,protocol,power))
#     # Envelope Fail (which is n, the y value of each point of the PSTH)
#     nsf_binning = []
#     for h in range(len(nsf_bins)):
#         if h>0:
#             nsf_binning.append(np.mean((nsf_bins[h],nsf_bins[h-1])))
#     nsf_step = time_trial/len(nsf_binning)
#     nsf_x = np.arange(0,time_trial,nsf_step)
#     nsf_nsmoothed = gaussian_filter1d(nsf_n, sigma=3.5)
#     ax4[2,1].plot(nsf_x, nsf_nsmoothed, color='k',alpha=0.9,linewidth=0.5)
#     ax4[2,1].set_title('Envelope')
#     ax4[2,1].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
#     ax4[2,1].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
#     ax4[2,1].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
#     ax4[2,1].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
#     ax4[2,1].set_ylabel('Number count')
#     ax4[2,1].set_xlabel('Time (s)')

###########
#   STIM  #
###########
# stim_success, stim_fail = [], []
# for t,l in new_stim_concat:
#     if t in stim_success_trial:
#         stim_success.append((t,l))
#     else:
#         stim_fail.append((t,l))

# stim_fail = np.asarray(stim_fail)
# stim_success = np.asarray(stim_success)

# fig5, ax5 = plt.subplots(3,2, figsize=(15,8), sharex=True)

# ## Plot Success
# if len(stim_success) > 0:
#     bv.scatter_lick(stim_success, ax=ax5[0,0])
#     ss_n,ss_bins,ss_patches = bv.PSTH_lick(stim_success, lentrial = time_trial, color='r', ax=ax5[1,0])
#     d=0.5
#     ax5[0,0].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
#     ax5[0,0].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
#     ax5[0,0].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
#     ax5[0,0].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
#     ax5[1,0].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
#     ax5[1,0].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
#     ax5[1,0].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
#     ax5[1,0].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
#     ax5[1,0].set_ylabel('Number count')
#     ax5[0,0].set_title('Success'.format(mouse,protocol,power))
#     # Envelope Success (which is n, the y value of each point of the PSTH)
#     ss_binning = []
#     for h in range(len(ss_bins)):
#         if h>0:
#             ss_binning.append(np.mean((ss_bins[h],ss_bins[h-1])))
#     ss_step = time_trial/len(ss_binning)
#     ss_x = np.arange(0,time_trial,ss_step)
#     ss_nsmoothed = gaussian_filter1d(ss_n, sigma=3.5)
#     ax5[2,0].plot(ss_x, ss_nsmoothed, color='k',alpha=0.9,linewidth=0.5)
#     ax5[2,0].set_title('Envelope')
#     ax5[2,0].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
#     ax5[2,0].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
#     ax5[2,0].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
#     ax5[2,0].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
#     ax5[2,0].set_ylabel('Number count')
#     ax5[2,0].set_xlabel('Time (s)')
## Plot Fail
# if len(stim_fail)>0:
#     bv.scatter_lick(stim_fail, ax=ax5[0,1])
#     sf_n,sf_bins,sf_patches = bv.PSTH_lick(stim_fail, lentrial = time_trial, color='r', ax=ax5[1,1])
#     d=0.5
#     ax5[0,1].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
#     ax5[0,1].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
#     ax5[0,1].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
#     ax5[0,1].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
#     ax5[1,1].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
#     ax5[1,1].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
#     ax5[1,1].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
#     ax5[1,1].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
#     ax5[1,1].set_ylabel('Number count')
#     ax5[0,1].set_title('Fail'.format(mouse,protocol,power))
    
#     # Envelope Fail (which is n, the y value of each point of the PSTH)
#     sf_binning = []
#     for h in range(len(sf_bins)):
#         if h>0:
#             sf_binning.append(np.mean((sf_bins[h],sf_bins[h-1])))
#     sf_step = time_trial/len(sf_binning)
#     sf_x = np.arange(0,time_trial,sf_step)
#     sf_nsmoothed = gaussian_filter1d(sf_n, sigma=3.5)
#     ax5[2,1].plot(sf_x, sf_nsmoothed, color='k',alpha=0.9,linewidth=0.5)
#     ax5[2,1].set_title('Envelope')
#     ax5[2,1].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
#     ax5[2,1].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
#     ax5[2,1].axvspan(2+op_t+d, 2+len_reward+op_t+d,color='r', alpha = 0.2) # Reward delivery
#     ax5[2,1].axvspan(2+len_reward+op_t+d,2+len_reward+op_t+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
#     ax5[2,1].set_ylabel('Number count')
#     ax5[2,1].set_xlabel('Time (s)')

# fig4.suptitle(f'{mouse}_{protocol}_{power}_NoStim_SuccvsFail')
# fig5.suptitle(f'{mouse}_{protocol}_{power}_Stim_SuccvsFail')

# fig4.savefig(fr"D:/F.LARENO.FACCINI/Preliminary Results/Behaviour/Group 11/Plots/PSTH/PSTH_Concatenate_{mouse}_{protocol}_{power}_NoStim_SuccvsFail.pdf")
# fig5.savefig(fr"D:/F.LARENO.FACCINI/Preliminary Results/Behaviour/Group 11/Plots/PSTH/PSTH_Concatenate_{mouse}_{protocol}_{power}_Stim_SuccvsFail.pdf")
plt.close('all')


# =============================================================================
# find time of lick > n_SD from noise        
# =============================================================================
# NO STIM
nostim_time_of_first_lick = []
nostim_time_of_last_lick = []
nostim_anticipation = []
# nostim_new_ant = []
for i in range(len(nsZ)):
    ns_convoluted_trial=nsZ[i]
    ns_list_of_index_above_sd=np.where(ns_convoluted_trial >= np.mean(ns_convoluted_trial[0:debut_win-1])+n_sd*np.std(ns_convoluted_trial[0:debut_win-1]))      
    ns_index_above_sd=ns_list_of_index_above_sd[0]
    ns_indice=bi.bisect(ns_index_above_sd, debut_win)  # trova l'indice del primo evento nella finestra d'interesse
    # ns_new_ant = np.where(nsZ[i] > np.mean(ns_convoluted_trial[0:debut_win-1])+n_sd*np.std(ns_convoluted_trial[0:debut_win-5]))
    if len(ns_index_above_sd)>0:
        nostim_time_of_first_lick=np.append(nostim_time_of_first_lick, ns_index_above_sd[ns_indice])
        nostim_anticipation=np.append(nostim_anticipation, float(ns_index_above_sd[ns_indice])*float(sampling)-(2+delay+OT)) # 2 sec for the protocol
    # if len(ns_new_ant)>0:
    #     nostim_new_ant = np.append(nostim_new_ant, (ns_new_ant[0][0]*float(sampling))-(2+delay+OT))
# STIM     
stim_time_of_first_lick = []
stim_time_of_last_lick = []
stim_anticipation = []
# stim_new_ant = []
for i in range(len(sZ)):
    s_convoluted_trial=sZ[i]
    s_list_of_index_above_sd=np.where(s_convoluted_trial >= np.mean(s_convoluted_trial[0:debut_win-1])+n_sd*np.std(s_convoluted_trial[0:debut_win-1]))      
    s_index_above_sd=s_list_of_index_above_sd[0]
    s_indice=bi.bisect(s_index_above_sd, debut_win)  # trova l'indice del primo evento nella finestra d'interesse
    # s_new_ant = np.where(sZ[i] > np.mean(ns_convoluted_trial[0:debut_win-1])+n_sd*np.std(ns_convoluted_trial[0:debut_win-5]))
    if len(s_index_above_sd)>0:
        stim_time_of_first_lick=np.append(stim_time_of_first_lick, s_index_above_sd[s_indice])
        stim_anticipation=np.append(stim_anticipation, float(s_index_above_sd[s_indice])*float(sampling)-(2+delay+OT)) # 2 sec for the protocol
    # if len(s_new_ant)>0:
    #     stim_new_ant = np.append(stim_new_ant, (s_new_ant[0][0]*float(sampling))-(2+delay+OT))

tot_control_trials = len(names)*nb_control_trials
tot_stim_trials = len(names)*nb_stim_trials

print (' ')  
print ('total trials: NOSTIM: {} --  STIM: {}'.format(tot_control_trials, tot_stim_trials))
print ('success: NOSTIM: {} --  STIM: {}'.format(len(nsZ), len(sZ)))
print ('success rate: NOSTIM: {} --  STIM: {}'.format(len(nsZ)/tot_control_trials, len(sZ)/tot_stim_trials))
print ('mean time first lick: NOSTIM: {} --  STIM: {}'.format(float(np.mean(nostim_time_of_first_lick))*sampling, float(np.mean(stim_time_of_first_lick))*sampling))
print ('mean anticipation: NOSTIM: {} --  STIM: {}'.format(np.mean(nostim_anticipation),np.mean(stim_anticipation)))
# print ('mean anticipation Fede: NOSTIM: {} -- STIM: {}'.format(np.mean(nostim_new_ant),np.mean(stim_new_ant)))

cols = ("Total Trials", "Successful Trials", "Success Rate", "Mean First Lick (s)","Mean Anticipation (s)")
# NO STIM
data = (tot_control_trials, len(nsZ), len(nsZ)/tot_control_trials, float(np.mean(nostim_time_of_first_lick))*sampling, np.mean(nostim_anticipation))
df = pd.DataFrame([data], columns=cols)
# STIM
s_data = (tot_stim_trials, len(sZ), len(sZ)/tot_stim_trials, float(np.mean(stim_time_of_first_lick))*sampling, np.mean(stim_anticipation))
s_df = pd.DataFrame([s_data], columns=cols)

with pd.ExcelWriter(fr"D:/F.LARENO.FACCINI/Preliminary Results/Behaviour/Group 14/Results/Results_{mouse}_{protocol}.xlsx", engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name = 'No Stim', index = False)
    s_df.to_excel(writer, sheet_name = 'Stim', index = False)




################################################


ns_ant = pd.DataFrame({'Anticipation': nostim_anticipation})
s_ant = pd.DataFrame({'Anticipation': stim_anticipation})

with pd.ExcelWriter(fr"D:/F.LARENO.FACCINI/Preliminary Results/Behaviour/Group 14/Results/Anticipation_{mouse}_{protocol}.xlsx", engine='openpyxl') as writer:
    ns_ant.to_excel(writer, sheet_name = 'No Stim', index = False)
    s_ant.to_excel(writer, sheet_name = 'Stim', index = False)
'''