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
    nostim, stim = bv.separate_by_condition(lick)    

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
