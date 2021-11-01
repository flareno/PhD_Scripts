# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:29:03 2021

@author: F.LARENO-FACCINI
"""

import numpy as np
import extrapy.Organize as og
import os
import scipy.signal as signal


def fixed_csv(mice, females, segment):
    old_sampling_rate = 20000
    new_sampling_rate = 1000
    ratio = int(old_sampling_rate/new_sampling_rate)

    protocols = {'NB': ['No Stim'],
               'P0': ['No Stim'],
               'P13': ['No Stim', 'Stim'],
               'P15': ['No Stim', 'Stim'],
               'P16': ['No Stim', 'Stim'],
               'P18': ['No Stim', 'Stim']}
    
    for protocol, value in protocols.items():        
        for cond in value:
            csv_all_mice = {}
            for group, topi in mice.items():
                for mouse in topi:
                    # print(group, mouse, protocol, cond)
                    
                    sex = 'Female' if mouse in females else 'Male'

                    basedir = fr'//equipe2-nas2/F.LARENO-FACCINI/BACKUP FEDE/Ephy/Group {group}/{mouse} (CM16-Buz - {sex})/Fixed Delay/{protocol}/{cond}'
                    
                    if os.path.exists(basedir):
                        print(f'{mouse} {protocol} {cond}, processing...')
                        files = og.file_list(basedir, False, '.rbf')
                        
                        lick_path = basedir.replace('Ephy', 'Behaviour').replace(f'/{cond}', '')
                        lick_path = lick_path+'/' + og.file_list(lick_path, no_extension=False, ext='.lick')[0]
                        licks,*_ = og.remove_empty_trials(lick_path,end_time='reward')
                        good_trials = np.unique(licks[:, 0])
                        
                        if cond == 'No Stim' and 'P1' in protocol:
                            good_trials = good_trials[good_trials < 31]
                            bad_trials = 30-len(good_trials)
                            delta = 1
                        elif cond == 'Stim':
                            good_trials = good_trials[good_trials > 30]
                            delta = 31
                            bad_trials = 30-len(good_trials)
                        else:
                            delta = 1
                            bad_trials = 60-len(good_trials)
                                                    
                        # if bad_trials >0:
                        if len(good_trials)>0:
                            for indx, file in enumerate(files):
                                if indx+delta in good_trials and 'McsRecording' in file:
                                    path = basedir + '//' + file
                                    
                                    sigs = np.fromfile(path, dtype='float').reshape(-1,16)
                                    sigs = sigs*1000 # from V to mV
                                    for ch in range(16):
                                        
                                        down_sig = signal.decimate(sigs[:,ch], ratio, n=4, zero_phase=True)
                                        if segment == 'First Half':
                                            cut_sig = down_sig[:int(2.5*new_sampling_rate)]
                                        else:
                                            cut_sig = down_sig[int(2.7*new_sampling_rate):]
                                            
                                        f,pxx = signal.welch(cut_sig, fs=new_sampling_rate, nperseg=new_sampling_rate)
                                        pxx = pxx[1:81] # save only from 1Hz - 80Hz
                                        
                                        if mouse not in csv_all_mice.keys():
                                            csv_all_mice[mouse] = pxx
                                        else:
                                            csv_all_mice[mouse] = np.vstack((csv_all_mice[mouse],pxx))
                    else:
                        print(f'{mouse} has no {protocol} {cond}!!')
            
            csv_all_mice['freqs'] = f[1:81]
            og.pickle_saving(savedir+f'/Database/{protocol}_{cond}_{segment}_Fixed_Good',csv_all_mice)




def random_csv(mice, females, segment):
    old_sampling_rate = 20000
    new_sampling_rate = 1000
    ratio = int(old_sampling_rate/new_sampling_rate)

    protocols = {'NB': ['No Stim'],
                'P0': ['No Stim'],
                'P13': ['No Stim', 'Stim'],
                'P15': ['No Stim', 'Stim'],
                'P16': ['No Stim', 'Stim'],
                'P18': ['No Stim', 'Stim']}
        

    for protocol, value in protocols.items():        
        for cond in value:
            csv_all_mice = {}
            for group, topi in mice.items():
                for mouse in topi:
                    # print(group, mouse, protocol, cond)
                    
                    sex = 'Female' if mouse in females else 'Male'

                    basedir = fr'//equipe2-nas2/F.LARENO-FACCINI/BACKUP FEDE/Ephy/Group {group}/{mouse} (CM16-Buz - {sex})/Random Delay/{protocol}/{cond}'
                    
                    if os.path.exists(basedir):
                        print(f'{mouse} {protocol} {cond}, processing...')
                        files = og.file_list(basedir, False, '.rbf')
                        
                        lick_path = basedir.replace('Ephy', 'Behaviour').replace(f'/{cond}', '')
                        lick_path = lick_path+'/' + og.file_list(lick_path, no_extension=False, ext='.lick')[0]
                        licks,*(random,ot) = og.remove_empty_trials(lick_path,end_time='reward',skip_last=True)
                        good_trials = np.unique(licks[:, 0])
                        random = np.array(random)
                        
                        
                        if cond == 'No Stim' and 'P1' in protocol:
                            good_trials = good_trials[good_trials < 31]
                            bad_trials = 30-len(good_trials)
                            delta = 1
                            max_len = 30
                            
                        elif cond == 'Stim':
                            good_trials = good_trials[good_trials > 30]
                            delta = 31
                            max_len = 29
                            bad_trials = 29-len(good_trials)
                        else:
                            delta = 1
                            max_len = 59
                            bad_trials = 59-len(good_trials)
                        
                        print('GOOD trials: ', len(good_trials))
                        # print('BAD trials: ', bad_trials)
                        
                        # if bad_trials >0:
                        if len(good_trials)>0:
                            for indx, file in enumerate(files):
                                # print(indx, delta+max_len)
                                if indx+delta in good_trials and 'McsRecording' in file and indx < max_len:
                                    # print(indx+delta)
                                    reward_off = 2+ot+(random[:,0][random[:,1] == indx+delta][0]/1000)+0.15
                                    # print(indx+delta, reward_off)
                                    
                                    t_start = reward_off-0.15
                                    t_stop = reward_off
                                    
                                    path = basedir + '//' + file
                                    
                                    sigs = np.fromfile(path, dtype='float').reshape(-1,16)
                                    sigs = sigs*1000 # from V to mV
                                
                                    for ch in range(16):
                                        
                                        down_sig = signal.decimate(sigs[:,ch], ratio, n=4, zero_phase=True)
                                        if segment == 'First Half':
                                            cut_sig = down_sig[:int(t_start*new_sampling_rate)]
                                        else:
                                            cut_sig = down_sig[int(t_stop*new_sampling_rate):]
                                        f,pxx = signal.welch(cut_sig, fs=new_sampling_rate, nperseg=new_sampling_rate)
                                        pxx = pxx[1:81] # save only from 1Hz - 80Hz

                                        if mouse not in csv_all_mice.keys():
                                            csv_all_mice[mouse] = pxx
                                        else:
                                            csv_all_mice[mouse] = np.vstack((csv_all_mice[mouse],pxx))
                                    # print(len(csv_all_mice), csv_all_mice[mouse].shape)
                    else:
                        print(f'{mouse} has no {protocol} {cond}!!')
            
            csv_all_mice['freqs'] = f[1:81]
            og.pickle_saving(savedir+f'/Database/{protocol}_{cond}_{segment}_Random_Good',csv_all_mice)



################################################################################
################################################################################
################################################################################

savedir = r'D:\F.LARENO.FACCINI\RESULTS\New Results\LFP\Welch_Half_Trial'

mice = {'14': (6401, 6402, 6409),
        '15': (173,  176),
        '16': (6924,),
        '17': (6456, 6457)}

females = (6409, 173, 176, 6456, 6457)

segments = ('First Half', 'Second Half')


for segment in segments:
    print(segment)
    fixed_csv(mice,females, segment=segment)
    # random_csv(mice,females, segment=segment)






