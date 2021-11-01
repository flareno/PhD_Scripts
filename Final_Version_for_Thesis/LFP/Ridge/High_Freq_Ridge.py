# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:07:13 2021

@author: F.LARENO-FACCINI
"""


def compute_power(mice, females, state='Good', sampling_rate=20000):
    delay_list = ('Fixed Delay','Random Delay')

    power_mean_dic = {'Fixed Delay': {}, 'Random Delay':{}}
    for group, topi in mice.items():
        for mouse in topi:
            print(mouse)
            protocols = {'P0': ['No Stim'],
                           'P13': ['No Stim', 'Stim'],
                           'P15': ['No Stim', 'Stim'],
                           'P16': ['No Stim', 'Stim'],
                           'P18': ['No Stim', 'Stim']}
            all_mice = (6401, 6402, 6409,173, 176,6924,6456, 6457)
            gender = 'Female' if mouse in females else 'Male'
    
            if int(group) < 15:
                del protocols['P0']
    
            for delay in delay_list:
                # if delay == 'Fixed':
                #     skip_last = False
                # else:
                #     skip_last = True
    
                for protocol, value in protocols.items():
                    if protocol == 'P0':
                        all_mice = (173, 176, 6924, 6456, 6457)

                    for cond in value:
                        if f'{protocol}_{cond}' not in power_mean_dic[delay].keys():
                            power_mean_dic[delay][f'{protocol}_{cond}'] = dict.fromkeys(all_mice)
    
                        print(mouse,protocol, cond)
                        basedir = fr'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\Ephy\Group {group}\{mouse} (CM16-Buz - {gender})\{delay}\{protocol}\{cond}'
                        list_files = og.file_list(basedir, no_extension=False, ext='.rbf')
    
                        lick_path = basedir.replace('Ephy', 'Behaviour').replace(f'\\{cond}', '')
                        lick_path = lick_path+'\\' + og.file_list(lick_path, no_extension=False, ext='.lick')[0]
                        licks, delays, ot = og.remove_empty_trials(lick_path,skip_last=False,end_time='reward')
                        good_trials = np.unique(licks[:, 0])
    
                        power = {}
                        if cond == 'No Stim' and 'P1' in protocol:
                            good_trials = good_trials[good_trials < 31]
                            delta = 1
                            bad_trials = 30-len(good_trials)
                        elif cond == 'Stim':
                            good_trials = good_trials[good_trials > 30]
                            delta = 31
                            bad_trials = 30-len(good_trials)
                        else:
                            delta = 1
                            bad_trials = 60-len(good_trials)
                        print(delay, 'Good Trials: ',len(good_trials),' Bad Trials: ', bad_trials)
    
                        # if cond == 'No Stim':
                        pox_values = len(good_trials)+bad_trials+delta
                        bad = [x for x in range(delta,pox_values) if x not in good_trials]
                        
                        
                        if state=='Good':
                            selected_trials = good_trials
                        else:
                            selected_trials = bad
                        
                        print(selected_trials)

                        if len(selected_trials)>0:
                        # if bad_trials>0:
                            for indx, file in enumerate(list_files):
                                if indx+delta in selected_trials and 'McsRecording' in file:
                                    path = basedir+'\\'+file
                                    sigs = np.fromfile(path, dtype=float).reshape(-1, 16)*1000 # to mV

                                    # print(indx+delta)
                                    
                                    for i in range(16):
                                        complex_map, map_times, freqs, tfr_sampling_rate = scalogram.compute_timefreq(sigs[:,i], sampling_rate, f_start=10, f_stop=30, delta_freq=1, f0=1)
                                        power_map = np.abs(complex_map)#**2
                                        power_map = power_map/np.sum(power_map)
                                        
                                        if f'Ch_{i}' not in power.keys():
                                            power[f'Ch_{i}'] = power_map
                                        else:
                                            power[f'Ch_{i}'] = np.dstack((power[f'Ch_{i}'],power_map))
    
                                        if indx+delta == max(selected_trials):
                                            if power[f'Ch_{i}'].ndim == 3:
                                                power[f'Ch_{i}'] = scalogram.ridge(np.mean(power[f'Ch_{i}'],axis=2))
                                            else:
                                                power[f'Ch_{i}'] = scalogram.ridge(power[f'Ch_{i}'])
    
                                    del(sigs, complex_map)
                                    print('shape power by mouse:', power['Ch_15'].shape)
                                
                                # else:
                                #     print(indx+delta, file)
                            
                            power_mean_dic[delay][f'{protocol}_{cond}'][mouse] = power
    
    return power_mean_dic, tfr_sampling_rate

def plot_single_mouse(power_mean_dic, band, state, save=False):
    for delay in power_mean_dic.keys():
        for session, topi in power_mean_dic[delay].items():
        
            if 'P0' not in session:
                for mouse, ch in topi.items():
                    if ch is not None:
                        mouse_avg = pd.Series([*ch.values()]).mean() # average all channels of a mouse
                        print(delay, session, mouse)
                        # BY CHANNEL
                        fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,5))
                        axes = axes.ravel()
                        
                        for i, (key, value) in enumerate(ch.items()):
                            time = np.arange(0,9, 9/len(value))
                
                            # print(session, mouse, i)
                            axes[i].plot(time,value)
                            axes[i].set_title(key)
                            axes[i].plot(time, mouse_avg, color='r',alpha=0.6,linewidth=0.6)
                            axes[i].axvspan(2.53, 2.68,color='r', alpha = 0.2) #Reward
                            axes[i].axvspan(0, 0.5,color='g', alpha = 0.2) # Cue1
                            axes[i].axvspan(1.5, 2,color='g', alpha = 0.2) #Cue2
                            axes[i].axvline(1.03, color='r', linestyle='--', alpha=0.2)
                    
                        fig.suptitle(f'{session} {mouse} {band}')
                        plt.show()
                        if save:
                            plt.savefig(fr'D:\F.LARENO.FACCINI\RESULTS\New Results\LFP\Ridge\Figures\{band}\{state} Trials\{delay}\Individual\By channel\{delay}_{session}_{mouse}_{band}.pdf')
                            plt.close()

                        
                        if 'No' in session:
                            # MOUSE AVERAGE
                            plt.figure()

                            if power_mean_dic[delay][session.replace('No Stim','Stim')][mouse] is not None:
                                stim_avg = pd.Series([*power_mean_dic[delay][session.replace('No Stim','Stim')][mouse].values()]).mean()
                                stim_std = np.std(np.array([*power_mean_dic[delay][session.replace('No Stim','Stim')][mouse].values()]),axis=0)
                                plt.plot(time, stim_avg,label='Stim', color='orange')
                                plt.fill_between(time, stim_avg+stim_std, stim_avg-stim_std, alpha=0.2, color='orange')

                            mouse_std = np.std(np.array([*ch.values()]),axis=0) # average all channels of a mouse
                            plt.plot(time, mouse_avg, label='No Stim', color='green')
                            plt.fill_between(time, mouse_avg+mouse_std, mouse_avg-mouse_std, alpha=0.2, color='green')
                            plt.axvspan(2.53, 2.68,color='r', alpha = 0.2) #Reward
                            plt.axvspan(0, 0.5,color='g', alpha = 0.2) # Cue1
                            plt.axvspan(1.5, 2,color='g', alpha = 0.2) #Cue2
                            plt.axvline(1.03, color='r', linestyle='--', alpha=0.2)
                            plt.title(f'{session} {mouse} {band} AVERAGE')
                            plt.legend()
                            
                            if save:
                                plt.savefig(fr'D:\F.LARENO.FACCINI\RESULTS\New Results\LFP\Ridge\Figures\{band}\{state} Trials\{delay}\Individual\Probe average\{delay}_{session}_{mouse}_{band}.pdf')
                                plt.close()
        
def plot_all_mice(power_mean_dic, band, state, save=False):
    for delay in power_mean_dic.keys():
        for session, topi in power_mean_dic[delay].items():
            all_mean_ns, all_mean_s = [], []
            
    
            if 'P0' not in session and 'No' in session:
                plt.figure()
                for mouse, ch in topi.items():
                    if ch is not None:
                        
                        print(delay, session, mouse)                    
                        # MOUSE AVERAGE
                        
                        
                        mouse_avg = pd.Series([*ch.values()]).mean() # average all channels of a mouse
                        all_mean_ns.append(mouse_avg)
                        if power_mean_dic[delay][session.replace('No Stim','Stim')][mouse] is not None:
                            stim_avg = pd.Series([*power_mean_dic[delay][session.replace('No Stim','Stim')][mouse].values()]).mean()
                            all_mean_s.append(stim_avg)
             
                all_mean_ns = np.array(all_mean_ns)
                print(all_mean_ns.shape)
        
                ns_sem = stats.sem(all_mean_ns,axis=0,nan_policy='omit')
                all_mean_ns = np.nanmean(all_mean_ns,axis=0)
        
                all_mean_s = np.array(all_mean_s)          
                s_sem = stats.sem(all_mean_s,axis=0,nan_policy='omit')
                all_mean_s = np.nanmean(all_mean_s,axis=0)
        
        
                time = np.arange(0,9, 9/len(all_mean_ns))
                plt.plot(time, all_mean_ns, label='No Stim', color='green')
                plt.fill_between(time, all_mean_ns+ns_sem, all_mean_ns-ns_sem, alpha=0.2, color='green')
        
                plt.plot(time, all_mean_s,label='Stim', color='orange')
                plt.fill_between(time, all_mean_s+s_sem, all_mean_s-s_sem, alpha=0.2, color='orange')
         
                plt.axvspan(2.53, 2.68,color='r', alpha = 0.2) #Reward
                plt.axvspan(0, 0.5,color='g', alpha = 0.2) # Cue1
                plt.axvspan(1.5, 2,color='g', alpha = 0.2) #Cue2
                plt.axvline(1.03, color='r', linestyle='--', alpha=0.2)
                plt.title(f'{session} {band} ALL MICE AVERAGED')
                plt.legend()
                
                if save:
                    plt.savefig(fr'D:\F.LARENO.FACCINI\RESULTS\New Results\LFP\Ridge\Figures\{band}\{state} Trials\{delay}\All averaged\Probe average\{delay}_{session}_{band}.pdf')
                    plt.close()


import extrapy.Scalogram as scalogram
import extrapy.Organize as og
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

savedir = r'D:\F.LARENO.FACCINI\RESULTS\New Results\LFP\Ridge'

mice = {
    '14': (6401, 6402, 6409),
    '15': (173, 176),
    '16': (6924,),
    '17': (6456, 6457)}

females = (6409, 173, 176, 6456, 6457)
state = 'Good'
band = 'Beta'

power_mean_dic, tfr_sampling_rate = compute_power(mice, females, state=state)

og.pickle_saving(savedir+fr'\Dataframe\{band}_{state}_Trials__tfrSR_{tfr_sampling_rate}',power_mean_dic)

# tfr_sampling_rate = 322.5806451612903
# power_mean_dic = og.pickle_loading(savedir+fr'\Dataframe\power_of_2_points\{band}_{state}_Trials__tfrSR_{tfr_sampling_rate}')

plot_single_mouse(power_mean_dic, band=band, state=state, save=True)

plot_all_mice(power_mean_dic, band=band, state=state, save=True)
