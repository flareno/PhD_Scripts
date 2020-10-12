# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:33:21 2020

@author: F.LARENO-FACCINI
"""

import matplotlib.pyplot as plt
import extrapy.Behaviour as bv
import glob
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.ndimage.filters import gaussian_filter1d

savedir = "D:\F.LARENO.FACCINI\Preliminary Results\Behaviour\Figures"
path = "D:\F.LARENO.FACCINI\Preliminary Results\Behaviour\Database\*.xlsx"
files = glob.glob(path)
#extract file name
files = [os.path.basename(i) for i in files]
mouse = [x.split("_")[-1] for x in files]
mouse = [x.replace(".xlsx","") for x in mouse]
ns = {}
ot = 0.03
d = 0.4
len_reward = 0.15
stims = {'P13': 0.8,'P15': 1.25, 'P16':1.16, 'P18':1}
topo = 176
delay = '900_400_400'

for indx,i in enumerate(files):
    file_path = fr"D:/F.LARENO.FACCINI/Preliminary Results/Behaviour/Database/"+i
    df_stim = pd.read_excel(file_path, sheet_name=f'Envelope_{delay}_Stim')
    df_nostim = pd.read_excel(file_path, sheet_name=f'Envelope_{delay}_NoStim')
    if indx == 0:
        bins = pd.read_excel(file_path, sheet_name=f'Bins_{delay}')
    keys = list(df_stim.keys())

    for e in keys:
        ns[mouse[indx]+'_'+e+'NoStim'] = df_nostim[e]
        ns[mouse[indx]+'_'+e+'Stim'] = df_stim[e]
bins = (np.asarray(bins))[:,0]
data = {}

protocols = [x.split("-")[-1] for x in ns.keys()]
protocols = set([x.split("N")[0] for x in protocols if 'N' in x])

for i in protocols:
    stim,nostim = [],[]
    for (key,values) in ns.items():
        if i in key and 'Stim' in key:
            norm_vs = (values/np.max(values))
            stim.append(norm_vs)
        if i in key and 'NoStim' in key:
            norm_vns = (values/np.max(values))
            nostim.append(norm_vns)
    data[i+'Stim'] = np.asarray(stim)
    data[i+'NoStim'] = np.asarray(nostim)
        
median_data,mad_data = {},{}
for k,v in data.items():
    median_data[k+'_median'] = np.median(v,axis=0)
    mad_data[k+'_mad'] = stats.median_abs_deviation(v,axis=0)
    
fig,ax = plt.subplots(2,2,sharex='col',sharey='row', figsize=(12,8))
ax = ax.ravel()

for ind,i in enumerate(protocols):
    for k,v in median_data.items():
        if 'NoStim' in k and i in k:
            print('NoStim: ',k)
            ns_convolved = bv.envelope(median_data[k],bins, sigma=2,ax=ax[ind],color='dodgerblue',alpha=0.7,y_label='Normalized Frequency of licks',label='Control' if ind == 0 else "")
            nsbins_convolved = gaussian_filter1d(mad_data[i+'NoStim_mad'], sigma=2)
            ax[ind].fill_between(bins[:-1],ns_convolved+nsbins_convolved,ns_convolved-nsbins_convolved,color='dodgerblue',alpha=0.2)
        elif i+'Stim' in k:
            print('Stim: ',k)
            n_convolved = bv.envelope(median_data[k],bins, sigma=2,ax=ax[ind],color='orange',alpha=0.7,y_label='Normalized Frequency of licks',label='Photostim' if ind == 0 else "")
            bins_convolved = gaussian_filter1d(mad_data[i+'Stim_mad'], sigma=2)
            ax[ind].fill_between(bins[:-1],n_convolved+bins_convolved,n_convolved-bins_convolved,color='orange',alpha=0.2)
    ax[ind].set_title(i)
    # Plotting details
    ax[ind].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
    ax[ind].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
    ax[ind].axvspan(2+ot+d, 2+len_reward+ot+d,color='r', alpha = 0.2) # Reward delivery
    ax[ind].axvspan(1.5, 1.5+next(x for y,x in stims.items() if y==i), color='skyblue', alpha = 0.3) # Opto Stim

    fig.legend()
    fig.suptitle(f'Median envelope all animals per protocol, delay: {delay}')
    # fig.suptitle(f'{topo}_Envelope per protocol')
fig.savefig(savedir+f'/All_animals_NoStim vs Stim_{delay}.pdf')
# plt.close()


