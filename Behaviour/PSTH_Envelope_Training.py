# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:56:24 2019

@author: F.LARENO-FACCINI

"""
import matplotlib.pyplot as plt
import extrapy.Behaviour as bv
import glob
import os

import time

startTime = time.time()


mouse = 6409
session = 'T9.1'
files = glob.glob(fr"D:\F.LARENO.FACCINI\Preliminary Results\Behaviour\Group 14\{mouse}\Training\{session}\*.lick")
names = [os.path.basename(x) for x in files]
names = [x.replace(".lick","") for x in names]

len_trial = 10
samp_period = 0.008
ot = 0.03
len_reward = 0.15
len_aspiration = 0.7

for file in (names):

    path = fr"D:\F.LARENO.FACCINI\Preliminary Results\Behaviour\Group 14\{mouse}\Training\{session}\{file}.lick"
    param = fr"D:\F.LARENO.FACCINI\Preliminary Results\Behaviour\Group 14\{mouse}\Training\{session}\{file}.param"
    
    # Load lick files and random delays
    licks = bv.load_lickfile(path)
    random = bv.extract_random_delay(param)
    
    # Separate trials by delay and Extract licks for the right delays
    delays, licks_by_delay = bv.separate_by_delay(random,licks, delay1=400, delay2=900)
 
    # Plot the lick events and the PSTH
    for key, values in licks_by_delay.items():
        if len(values) > 0:
            fig, ax = plt.subplots(2, sharex=True)
            bv.scatter_lick(values, ax=ax[0])
            n,bins,patches = bv.psth_lick(values, lentrial=len_trial,density=True, samp_period=samp_period, color='r', ax=ax[1])
            ax[0].set_title('{} // Delay of: {}ms'.format(file, key))
            
            if key == '400' or key == '400_400' or key == '900_400' or key == '400_400_400' or key == '900_400_400':
                d=0.4
                c=0.9
            elif key == '900' or key == '400_900' or key == '900_900' or key == '900_900_900' or key == '400_900_900':
                d=0.9
                c=0.4  
                
            # Plot the envelope of the PSTH          
            bv.envelope(n,bins,ax=ax[1],sigma=3) 

            # Plotting details
            ax[0].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
            ax[0].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
            ax[0].axvspan(2+ot+d, 2+len_reward+ot+d,color='r', alpha = 0.2) # Reward delivery
            ax[0].axvspan(2+len_reward+ot+d,2+len_reward+ot+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
            ax[1].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
            ax[1].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
            ax[1].axvspan(2+ot+d, 2+len_reward+ot+d,color='r', alpha = 0.2) # Reward delivery
            ax[1].axvspan(2+len_reward+ot+d,2+len_reward+ot+d+len_aspiration,color='purple', alpha=0.2) # Aspiration
            ax[1].set_ylabel('Density')
            ax[1].axvline(2+ot+c, linestyle='--', color='k', linewidth=0.7, alpha=0.5)

            # plt.close('all')
            
print ('The script took {0} seconds!'.format(time.time() - startTime))
