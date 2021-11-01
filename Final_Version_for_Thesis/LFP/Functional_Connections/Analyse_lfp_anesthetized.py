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
import pandas as pd

def all_together():
    basedir = r'D:\F.LARENO.FACCINI\Preliminary Results\Ephy\Coordinate Hunting\All Together'
    dirs = os.listdir(basedir)
    freq_high = 60
    freq_low = 0.1
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
        
        
        plt.figure()    
        plt.plot(time,all_)
        plt.fill_between(time,all_+sd_all,all_-sd_all,alpha=0.5,interpolate=True)
        
        
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
            plt.axvspan(6,7,color='skyblue',alpha=0.4)
    
        plt.ylim(-0.5,0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (mV)')
        plt.title(f'{dir_} {protocol} LPF: {freq_high}Hz  HPF: {freq_low}')
        # plt.savefig(fr'D:\F.LARENO.FACCINI\RESULTS\New Results\LFP\Mapping of Cerebellum\All_Together\{dir_}_{protocol}.pdf')
        # plt.close()







def peak_by_mouse(conditions, protocols, freq_high, freq_low=0.1, sampling_rate=20000):
    all_peaks = []
    
    for condition in conditions:
        for protocol in protocols:
            mice = [852,854,853,855,858]
            if protocol == 'P13':
                del(mice[:2])
                
            for mouse in mice:
                print(condition, protocol, mouse)
            
                basedir = fr'D:\F.LARENO.FACCINI\Preliminary Results\Ephy\Coordinate Hunting\{mouse}\RBF\{condition}\{protocol}'
            
                files = og.file_list(basedir,False,'.rbf')
                # print(len(files))
                if len(files) != 0:
                    all_ = []
                    for file in files:
                        sigs = np.fromfile(basedir+'\\'+file).reshape(-1,16)
                        temp__ = np.mean(sigs,axis=1)
                        temp__ = filters.bandpass_filter(temp__,freq_low=freq_low,freq_high=freq_high)
                        all_.append(temp__)
                     
                    len_sig = len(temp__)/sampling_rate
                    time = np.arange(0,len_sig,1/sampling_rate)
                    
                    all_ = np.array(all_) * 1000 # V to mV
                    # print(all_.shape)            
                    
                    sd_all = stats.sem(all_,axis=0)
                    all_ = np.mean(all_,axis=0)    
                    
                    if protocol == 'P13':
                        stop = 0.8
                        stim_on = 6
                        stim_off = 6.8
                    elif protocol == 'P19':
                        stop = 1
                        stim_on = 6
                        stim_off = 7
                    else:
                        stop = 4
                        stim_on = 4
                        stim_off = 8
                        
                    baseline = (int(sampling_rate*0),int(sampling_rate*(0+stop)))
                    peak = (int(sampling_rate*stim_on), int(sampling_rate*stim_off))
                    after_peak = (int(sampling_rate*stim_off), int(sampling_rate*(stim_off+stop)))
                    tail_baseline = (int(sampling_rate*10),int(sampling_rate*(10+stop)))
                    
                    all_peaks.append([mouse, condition, protocol, np.min(all_[baseline[0]:baseline[1]]), np.min(all_[peak[0]:peak[1]]), np.min(all_[after_peak[0]:after_peak[1]]), np.min(all_[tail_baseline[0]:tail_baseline[1]])])
                    
                    # plt.figure()    
                    # plt.plot(time,all_)
                    # plt.fill_between(time,all_+sd_all,all_-sd_all,alpha=0.5)
                    
                    # if protocol == 'P20':
                    #     #Time vector of the whole trace
                    #     stim_time = 4
                    #     tot_stim = 4
                    #     stim_duration = 0.4
                    #     #Stim vector for the whole trace
                    #     stim_vector = np.arange(stim_time, stim_time+tot_stim,1)
                    
                    #     for stim in stim_vector:
                    #         plt.axvspan(stim,stim+stim_duration,color='b',alpha=0.2)
                    # else:
                    #     plt.axvspan(6,6.8 if protocol=='P13' else 7,color='skyblue',alpha=0.4)
                
                    
                    # plt.xlabel('Time (s)')
                    # plt.ylabel('Amplitude (mV)')
                    # plt.title(f'{mouse} {condition} {protocol} LPF: {freq_high}Hz  HPF: {freq_low}')
                    # plt.savefig(fr'D:\F.LARENO.FACCINI\RESULTS\New Results\LFP\Mapping of Cerebellum\Individual_Mouse\{condition}_{mouse}_{protocol}.pdf')
                    # plt.close()
    
    cols = ['Mouse', 'Location', 'Protocol', 'Peak_Baseline', 'Peak_Stim', 'Peak_After', 'Peak_Tail']
    return pd.DataFrame(all_peaks,columns=cols,index=None)



savedir = r'D:\F.LARENO.FACCINI\RESULTS\New Results\LFP\Mapping of Cerebellum'
locations = ('CrusI Lat', 'CrusI Med', 'CrusII Lat', 'CrusII Med', 'VI', 'VII')
protocols = ('P13', 'P19', 'P20')
freq_high = 60
freq_low = 0.1
sampling_rate = 20000

# all_together()
# all_peaks = peak_by_mouse(locations, protocols, freq_high)

# og.pickle_saving(savedir+'\\df_all_peaks', all_peaks)
all_peaks = og.pickle_loading(savedir+'\\df_all_peaks')

# =============================================================================
#  STATS
# =============================================================================
for location in locations:
    for protocol in protocols:
        
        new_df = all_peaks[(all_peaks['Location'] == location) & (all_peaks['Protocol'] == protocol)]
        W,p_value = stats.wilcoxon(new_df['Peak_Baseline'],new_df['Peak_Stim'],alternative='greater')
        if p_value < 0.05:
            print(location, protocol, 'p-value PEAK: ', p_value)

        # print(location,protocol)
        # print(new_df.describe())
        # print(new_df.Peak_Baseline.sem())
        # print(new_df.Peak_Stim.sem())
# =============================================================================
#  BOXPLOT
# =============================================================================

plot_df = all_peaks[(all_peaks['Location'] == 'CrusI Lat') & (all_peaks['Protocol'] == 'P19')]

plot_df.boxplot(column=['Peak_Baseline', 'Peak_Stim'])
plt.ylabel('Peak Amplitude (mV)')
plt.title('Baseline vs. Stimulation of CrusI (p-value = 0.031)')

