# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 13:00:11 2021

@author: F.LARENO-FACCINI
"""
import extrapy.Organize as og
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import extrapy.Filters as filters


basedir = r'D:\F.LARENO.FACCINI\Preliminary Results\Ephy\Coordinate Hunting\All Together'
dirs = os.listdir(basedir)
freq_high = 1000
freq_low = 100
protocol = 'P19'

if protocol == 'P13':
    del(dirs[4])

for dir_ in dirs:
    new_dir = basedir+'\\'+dir_+f'\\{protocol}'
    files = og.file_list(new_dir,False,'.rbf')
    sampling_rate = 20000
    
    all_ = []
    for file in files:
        sigs = np.fromfile(new_dir+'\\'+file).reshape(-1,16)
        temp__ = np.mean(sigs,axis=1)
        temp__ = filters.bandpass_filter(temp__,freq_low=freq_low,freq_high=freq_high)
        all_.append(temp__)
     
    len_sig = len(temp__)/sampling_rate
    time = np.arange(0,len_sig,1/sampling_rate)
        
    all_ = np.array(all_) * 1000 # V to mV
    sd_all = stats.sem(all_,axis=0)
    all_ = np.mean(all_,axis=0)    
    
    
    z_all = all_[int(6*sampling_rate-10000):int(7*sampling_rate+15000)]
    semz = sd_all[int(6*sampling_rate-10000):int(7*sampling_rate+15000)]


    duration = 1./sampling_rate * len(z_all)
    time = np.arange(0, duration, 1./sampling_rate)

    plt.figure()    
    plt.plot(time,z_all)
    
    plt.plot(time,z_all+semz, color='r')
    plt.plot(time,z_all-semz, color='r')
    
    if protocol == 'P20':
        #Time vector of the whole trace
        stim_time = 4
        tot_stim = 4
        stim_duration = 0.4
        #Stim vector for the whole trace
        stim_vector = np.arange(stim_time, stim_time+tot_stim,1)
    
        for stim in stim_vector:
            plt.axvspan(stim,stim+stim_duration,color='b',alpha=0.2)
    else:
        plt.axvspan(0.5,1.5,color='skyblue',alpha=0.4)

    plt.ylim(-0.5,0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.title(f'{dir_} {protocol} LPF: {freq_high}Hz  HPF: {freq_low}')
    plt.xticks(np.arange(0,((7+0.75)-(6-0.5)),0.5),labels=np.arange((6-0.5),(7+0.75),0.5))
    # plt.set_xticklabels(np.arange((6-0.5),(7+0.75),0.5))

    # plt.savefig(fr'D:\F.LARENO.FACCINI\Tesi\Figures\Mapping the cerebellum\{dir_}_{protocol}_zoomed.pdf')
    # plt.close()
