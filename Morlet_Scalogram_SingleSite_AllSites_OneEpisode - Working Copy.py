# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:11:22 2019

@author: F.LARENO-FACCINI
"""

#from compute_timefreq import compute_timefreq
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.stats import sem
from extrapy import Scalogram, Filters


'''
In here I am:
    - computing the Morlet Scalogram                  |
    - extracting the power values                     | --> All of this For each single site of each recording, individually 
    - appending these power values in a Excel file    |
'''
    
### Parameters
protocol = 'P15'
mouse = 6401
condition = 'Stim'

#Sampling rate (in Hertz) and start time (in seconds) and channel
sampling_rate= 20000 
t_start = 0 

#Freq window for Morlet Wavelet scalo (in Hz)
f_start = 1
f_stop = 20
#    f_start = 32
#    f_stop = 80.

#Freq borders for filtered plot (in Hz)
freq_low=0.1
freq_high=45
   
#Time of light stim
stim_start = 1.5
stim_end = 2.75

####### iterate trhough files ######

#Import all the files, by creating a list of them
files = glob.glob(fr'\\equipe2-nas1\F.LARENO-FACCINI\BACKUP FEDE\Ephy\6401 (CM16-Buz - Male)\Experiment\Fixed Delay\{protocol}\{condition}\*.rbf')
names = [os.path.basename(x) for x in files]
savedir = (fr'D:/F.LARENO.FACCINI/Preliminary Results/Ephy/6401 (CM16-Buz - Male)/Experiment/Fixed Delay/{protocol}/{condition}/')


#    cmap = plt.cm.get_cmap('viridis')
cmap = ['black', 'red', 'blue', 'seagreen', 'gold', 'purple', 'deepskyblue', 'lightsalmon', 'hotpink', 'yellow', 'cyan', 'magenta']

#Remove the extension from the name in the list
for x in range(len(names)):
    names[x] = names[x].replace(".rbf","")
   
if len(names) == 0:
    print('This is not the directory you are looking for')
   
else:
    temp__= []          
    
    for f in range(len(names)):

        print (names[f])
        #loop to iterate the files
       
        #Location of the file
        path =fr'\\equipe2-nas1\F.LARENO-FACCINI\BACKUP FEDE\Ephy\6401 (CM16-Buz - Male)\Experiment\Fixed Delay\{protocol}\{condition}\{names[f]}.rbf'
        sigs = np.fromfile(path, dtype='float64').reshape(-1,16)
        
        
        for j in range(16):
            temp__.append(np.array(sigs[:,[j]]))
    
        avg = np.mean(temp__, axis=0).ravel()
        se = sem(temp__, axis=0).ravel()

        #Define the signal, its length in seconds and a time vector
        duration = 1./sampling_rate * len(avg)
        sig_times = np.arange(0, duration, 1./sampling_rate)
        
        if len(sig_times) > len(avg):
            sig_times = sig_times[:-1]
        
        notch = Filters.notch_filter(avg)

        
        #Fig 1 : the raw signal
        filt_notch = Filters.bandpass_filter(avg,freq_low=freq_low,freq_high=freq_high)
        
        #Fig 1 : the raw signal
        fig, ax = plt.subplots(2,1, sharex=True, figsize=(12,8))
        ax[0].set_title('Here is the averaged signal')
        ax[0].plot(sig_times, avg, linewidth=0.3, alpha=0.4,  label='Raw signal')
        ax[0].fill_between(sig_times,avg,avg+se, color='skyblue', alpha=0.3)
        ax[0].fill_between(sig_times,avg,avg-se, color='skyblue', alpha=0.3)
        ax[0].plot(sig_times,filt_notch, linewidth=1.2,color='orange', label='Filtered signal ({}-{}Hz)'.format(freq_low,freq_high))
        ax[0].fill_between(sig_times,filt_notch,filt_notch+se, color='orange', alpha=0.2)
        ax[0].fill_between(sig_times,filt_notch,filt_notch-se, color='orange', alpha=0.2)
        ax[0].axvspan(0, 0.5,color='g', alpha = 0.2)
        ax[0].axvspan(1.5, 2,color='g', alpha = 0.2)
        ax[0].axvspan(2.5, 2.55,color='r', alpha = 0.2)

        #############################################
        ### PLOTTING THE NON-NOTCH FILTERED TRACE ###
        #############################################
        
#        ax[0].plot(sig_times,filter_signal(avg,freq_low=freq_low,freq_high=freq_high), linewidth=0.5,color='k', alpha=0.9, label='BP-Filtered signal ({}-{}Hz)'.format(freq_low,freq_high))

        #Fig 2 : Timefreq
        ax[1].set_ylabel('Freq (Hz)')
        ax[1].set_title('Scalogram')
        ax[1].set_xlabel('Time (s)')
        ax[1].axvspan(0, 0.5,color='g', alpha = 0.2)
        ax[1].axvspan(1.5, 2,color='g', alpha = 0.2)
        ax[1].axvspan(2.5, 2.55,color='r', alpha = 0.2)

        complex_map, map_times, freqs, tfr_sampling_rate = Scalogram.compute_timefreq(avg, sampling_rate, f_start, f_stop, delta_freq=0.1, nb_freq=None,
                        f0=2.5,  normalisation = 0, t_start=t_start)
        
        ampl_map = np.abs(complex_map) # the amplitude map (module)
        phase_map = np.angle(complex_map) # the phase
        
        delta_freq = freqs[1] - freqs[0]
        extent = (map_times[0], map_times[-1], freqs[0]-delta_freq/2., freqs[-1]+delta_freq/2.)
        
        scalo = ax[1].imshow(ampl_map.transpose(), interpolation='nearest', 
                            origin ='lower', aspect = 'auto', extent = extent, cmap='viridis')


        ax[0].axvspan(stim_start, stim_end,color='b', alpha = 0.2)
        ax[1].axvspan(stim_start, stim_end,color='b', alpha = 0.2)
        ax[0].set_title(f'{mouse}_{names[f]}_{condition}_ProbeAveraged')
        ax[0].legend()

        # fig.colorbar(scalo)
        # plt.savefig(savedir + f'{names[f]}_{condition}_ProbeAveraged.png')
        # plt.close()
        
        #---------------------- Extracting power values --------------------------------
        
        start_before = int(1.*tfr_sampling_rate)
        end_before = int(1.5*tfr_sampling_rate)
        start_during = int(1.8*tfr_sampling_rate)
        end_during = int(2.3*tfr_sampling_rate)
        start_after = int(2.3*tfr_sampling_rate)
        end_after = int(2.8*tfr_sampling_rate)
        
        before = ampl_map.transpose()[:,start_before:end_before]
        during = ampl_map.transpose()[:,start_during:end_during]
        after = ampl_map.transpose()[:,start_after:end_after]

        avg_before = np.true_divide(before.sum(1),(before!=0).sum(1))
        avg_during = np.true_divide(during.sum(1),(during!=0).sum(1))
        avg_after = np.true_divide(after.sum(1),(after!=0).sum(1))
                
#                print(after.shape)
#                print(len(avg_after))
        #------------------------ Writing to the Excel file --------------------------- 
#    #            print(names[f],i)
#                  
#                a = np.vstack((avg_before))#.transpose()
##                b = np.vstack((avg_during))#.transpose()
##                c = np.vstack((avg_after))#.transpose()
#                
#                if f+i==0: 
#                    df1 = pd.DataFrame(a,columns=['%s_%s'%(names[f],i)])
##                    df2 = pd.DataFrame(b,columns=['%s_%s'%(names[f],i)])
##                    df3 = pd.DataFrame(c,columns=['%s_%s'%(names[f],i)])
#    
#                else:                
#                    df1['%s_%s'%(names[f],i)] = a
##                    df2['%s_%s'%(names[f],i)] = b
##                    df3['%s_%s'%(names[f],i)] = c
#    
#    
#                powerdir = (r'D:/F.LARENO.FACCINI/Preliminary Results/Ephy/Bands')
#                  
##                with pd.ExcelWriter('{}/{}_power_values_valve_{}_NoStim_{}.xlsx'.format(powerdir,mouse,protocol,state), engine='openpyxl') as writer:
#                with pd.ExcelWriter('{}/average_band.xlsx'.format(powerdir), engine='openpyxl') as writer:
#                    df1.to_excel(writer, sheet_name='Before (500 ms)')
##                    df2.to_excel(writer, sheet_name='During (500 ms)')
##                    df3.to_excel(writer, sheet_name='After (500 ms)')
#    

