# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:43:43 2021

@author: F.LARENO-FACCINI
"""

def power_extraction(mice,females):
    delay_list = ('Fixed Delay','Random Delay')
    # delay_list = ('Random Delay',)#'Random Delay')
    sampling_rate = 20000
    power_mean_dic = {}
    for group, topi in mice.items():
        for mouse in topi:
            print(mouse)
            protocols = {'P0': ['No Stim'],
                          'P13': ['No Stim', 'Stim'],
                          'P15': ['No Stim', 'Stim'],
                          'P16': ['No Stim', 'Stim'],
                          'P18': ['No Stim', 'Stim']}

            gender = 'Female' if mouse in females else 'Male'

            if int(group) < 15:
                del protocols['P0']

            for delay in delay_list:
                if 'Fixed' in delay:
                    skip_last = False
                else:
                    skip_last = True

                for protocol, value in protocols.items():
                    for cond in value:
                        if f'{protocol}_{cond}' not in power_mean_dic.keys():
                            power_mean_dic[f'{protocol}_{cond}'] = {}

                        print(protocol, cond)

                        basedir = fr'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\Ephy\Group {group}\{mouse} (CM16-Buz - {gender})\{delay}\{protocol}\{cond}'
                        list_files = og.file_list(basedir, no_extension=False, ext='.rbf')

                        lick_path = basedir.replace('Ephy', 'Behaviour').replace(f'\\{cond}', '')
                        lick_path = lick_path+'\\' + og.file_list(lick_path, no_extension=False, ext='.lick')[0]
                        licks, delays, ot = og.remove_empty_trials(lick_path,skip_last=skip_last,end_time='reward')#,return_delays=False)

                        licks_dic = {delay: licks}

                        power = {}

                        for time, lick_data in licks_dic.items():
                            good_trials = np.unique(lick_data[:, 0])

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
                            print(time, 'Good Trials: ',len(good_trials),' Bad Trials: ', bad_trials)
                           
                            pox_values = len(good_trials)+bad_trials+delta
                            bad = [x for x in range(delta,pox_values) if x not in good_trials]
                            
                            if state=='Good':
                                selected_trials = good_trials
                            else:
                                selected_trials = bad
                            
                            print(selected_trials)
    
                            if len(selected_trials)>0:
                                for indx, file in enumerate(list_files):
                                    if indx+delta in selected_trials and 'McsRecording' in file:
                                        path = basedir+'\\'+file
                                        sigs = np.fromfile(path, dtype=float).reshape(-1, 16)
                                        # print(indx+delta)
                                        temp__ = np.mean(sigs, axis=1)*1000 # to mV
                                        complex_map, map_times, freqs, tfr_sampling_rate = scalogram.compute_timefreq(temp__, sampling_rate, f_start=1.5, f_stop=4, delta_freq=1, f0=1)
                                        power_map = np.abs(complex_map)#**2
                                        power_map = power_map/np.sum(power_map) # NORMALIZE
                                        del(sigs, complex_map)
                                        # print('Before', power_map.shape)
                                        
                                        # if time == 'Random Delay':
                                            # current_delay = 2+delays[indx][0]/1000+ot
                                            # # real_index_reward = int(current_delay*tfr_sampling_rate)
                                            # # min_index_reward = int((2.4+ot)*tfr_sampling_rate)
                                            # print(current_delay)
                                            # to_cut = int(((2.9+ot)*tfr_sampling_rate)-((2.4+ot)*tfr_sampling_rate))
                                            
                                            # if current_delay > 2.6:
                                            #     power_map = power_map[to_cut:,:]
                                            # else:
                                            #     power_map = power_map[:-to_cut,:]
                                            # # print('After',power_map.shape)
                                            
                                        if f'{protocol}_{cond}_{time}' not in power.keys():
                                            power[f'{protocol}_{cond}_{time}'] = power_map
                                        else:
                                            power[f'{protocol}_{cond}_{time}'] = np.dstack((
                                                power[f'{protocol}_{cond}_{time}'], power_map))
                                    
                                print('shape power by mouse:', power[f'{protocol}_{cond}_{time}'].shape)
    
    
    
                                if f'{protocol}_{cond}_{time}' in power:
                                    if f'{time}_{mouse}' not in power_mean_dic[f'{protocol}_{cond}'].keys():
                                        if len(power[f'{protocol}_{cond}_{time}'].shape) == 3:
                                            power_mean_dic[f'{protocol}_{cond}'][f'{time}_{mouse}'] = np.array(np.mean(power[f'{protocol}_{cond}_{time}'],axis=2))
                                        else:
                                            power_mean_dic[f'{protocol}_{cond}'][f'{time}_{mouse}'] = np.array(power[f'{protocol}_{cond}_{time}'])
                                   
                                    else:
                                        if len(power[f'{protocol}_{cond}_{time}'].shape) == 3:
                                            mean_mouse = np.mean(power[f'{protocol}_{cond}_{time}'],axis=2)
                                        else:
                                            mean_mouse = power[f'{protocol}_{cond}_{time}']
                                        power_mean_dic[f'{protocol}_{cond}'][f'{time}_{mouse}'] = np.dstack((
                                            power_mean_dic[f'{protocol}_{cond}'][f'{time}_{mouse}'], mean_mouse))
    
                                    print(power_mean_dic[f'{protocol}_{cond}'][f'{time}_{mouse}'].shape)
   
    return power_mean_dic,tfr_sampling_rate




def plot_maker(power_mean_dic, state, band, tfr_sampling_rate, save=False):
    # for session in power_mean_dic.keys():
    #     if 'P1' in session and 'No' in session:
    #         for delay, data in power_mean_dic[session].items():
    #             # print(session,delay)
                
    delays = ('Fixed Delay', 'Random Delay')
    for delay in delays:
        for session in power_mean_dic.keys():
            if 'P1' in session and 'No' in session:
                temp_ns, temp_s = [], []
                for key,data in power_mean_dic[session].items():
                    if delay in key:
                        print(session, key)

                        start = int(0.5*tfr_sampling_rate)
                        stop = int(8.5*tfr_sampling_rate)
                        time = np.arange(0.5,8.5, 8/len(data[:,0][start:stop]))
                        # if delay == 'Random Delay':
                        #     time-2.43
            
                        # loop over individual mouse
                        # to compute the ridge of each mouse (on the average power of the mouse)
                        ns_ridge = scalogram.ridge(data)
                        temp_ns.append(ns_ridge)
                        print(len(temp_ns))
                        # plt.plot(time,ns_ridge[start:stop],color='green',alpha=0.2)
                        try:
                            s_ridge = scalogram.ridge(power_mean_dic[session.replace('No ','')][key])
                            temp_s.append(s_ridge)
                            # plt.plot(time,s_ridge[start:stop],color='orange',alpha=0.2)
                        except:
                            pass
            
                
                new_ns = np.nanmean(np.asarray(temp_ns),axis=0)
                ns_sem = stats.sem(np.asarray(temp_ns),axis=0, nan_policy='omit')
                new_s = np.nanmean(np.asarray(temp_s),axis=0)
                s_sem = stats.sem(np.asarray(temp_s),axis=0, nan_policy='omit')
                
                plt.figure()
                plt.plot(time,new_ns[start:stop],color='green',label='Control')
                plt.plot(time,new_s[start:stop],color='orange',label='Stim')
                plt.fill_between(time,new_ns[start:stop]+ns_sem[start:stop],new_ns[start:stop]-ns_sem[start:stop],interpolate=True, color='green',alpha=0.2)
                plt.fill_between(time,new_s[start:stop]+s_sem[start:stop],new_s[start:stop]-s_sem[start:stop],interpolate=True, color='orange',alpha=0.2)
                plt.legend()
                
                if delay == 'Fixed Delay':
                    plt.axvspan(2.53, 2.68,color='r', alpha = 0.2) #Reward
                else:
                    plt.axvline(2.43,color='r',linestyle='--', alpha = 0.2) #Reward
                    plt.axvline(2.93,color='r',linestyle='--', alpha = 0.2) #Reward
    
                plt.axvspan(0, 0.5,color='g', alpha = 0.2) # Cue1
                plt.axvspan(1.5, 2,color='g', alpha = 0.2) #Cue2
                if 'P13' in session:
                    stim_len = 0.8
                elif 'P15' in session:
                    stim_len = 1.25
                elif 'P16' in session:
                    stim_len = 1.16
                else:
                    stim_len = 1
                
                plt.axvspan(1.5,1.5+stim_len, color='skyblue', alpha=0.3)
                
                plt.title(f'{session.replace("No ","")}, {delay}, {band} {state} TRIALS')
                if save:
                    plt.savefig(savedir+fr'\Figures\{band}\{state} Trials\{delay}\{session.replace("No ","")}_{delay}_Ridge_{band}_{state}.pdf')
           


def plot_sex(power_mean_dic,females, state, band, tfr_sampling_rate, save=False):
    delays = ('Fixed Delay', 'Random Delay')
    fem = (6409, 173, 176, 6456, 6457)
    mal = (6401,6402,6924)
    sexes = ('females', 'males')
    
    for delay in delays:
        for session in power_mean_dic.keys():
            
            if 'P1' in session and 'No' in session:
                fig, ax = plt.subplots(2,1,sharex=True, sharey=True)
                for i, sex in enumerate(sexes):
                    print(sex)
                    temp_ns, temp_s = [], []
                    chosen=fem if sex == 'females' else mal
                    for key,data in power_mean_dic[session].items():
                        if delay in key and int(key.split('_')[-1])  in chosen:
                            print(session, key)
                            # print(session,delay)
                            
                            
                            start = int(0.5*tfr_sampling_rate)
                            stop = int(8.5*tfr_sampling_rate)
                            time = np.arange(0.5,8.5, 8/len(data[:,0][start:stop]))
                            # if delay == 'Random Delay':
                            #     time = time-2.43
        
                            # loop over individual mouse
                            # to compute the ridge of each mouse (on the average power of the mouse)
                            ns_ridge = scalogram.ridge(data)
                            temp_ns.append(ns_ridge)
                            print(len(temp_ns))
                            # plt.plot(time,ns_ridge[start:stop],color='green',alpha=0.2)
                            try:
                                s_ridge = scalogram.ridge(power_mean_dic[session.replace('No ','')][key])
                                temp_s.append(s_ridge)
                                # plt.plot(time,s_ridge[start:stop],color='orange',alpha=0.2)
                            except:
                                pass
                    
                    new_ns = np.nanmean(np.asarray(temp_ns),axis=0)
                    ns_sem = stats.sem(np.asarray(temp_ns),axis=0,nan_policy='omit')
                    new_s = np.nanmean(np.asarray(temp_s),axis=0)
                    s_sem = stats.sem(np.asarray(temp_s),axis=0,nan_policy='omit')
                    # plt.figure()                
                    ax[i].plot(time,new_ns[start:stop],color='green',label='Control')
                    ax[i].plot(time,new_s[start:stop],color='orange',label='Stim')
                    ax[i].fill_between(time,new_ns[start:stop]+ns_sem[start:stop],new_ns[start:stop]-ns_sem[start:stop],interpolate=True, color='green',alpha=0.2)
                    ax[i].fill_between(time,new_s[start:stop]+s_sem[start:stop],new_s[start:stop]-s_sem[start:stop],interpolate=True, color='orange',alpha=0.2)
                    ax[i].set_title(sex)
                    if delay == 'Fixed Delay':
                        ax[i].axvspan(2.53, 2.68,color='r', alpha = 0.2) #Reward
                    else:
                        ax[i].axvline(2.43,color='r',linestyle='--', alpha = 0.2) #Reward
                        ax[i].axvline(2.93,color='r',linestyle='--', alpha = 0.2) #Reward

                    ax[i].axvspan(0, 0.5,color='g', alpha = 0.2) # Cue1
                    ax[i].axvspan(1.5, 2,color='g', alpha = 0.2) #Cue2
                    if 'P13' in session:
                        stim_len = 0.8
                    elif 'P15' in session:
                        stim_len = 1.25
                    elif 'P16' in session:
                        stim_len = 1.16
                    else:
                        stim_len = 1
                    
                    ax[i].axvspan(1.5,1.5+stim_len, color='skyblue', alpha=0.3)
                   
                
                fig.legend()
                fig.suptitle(f'{session.replace("No ","")}, {delay}, {band} {state} TRIALS')
                if save:
                    plt.savefig(savedir+fr'\Figures\{band}\{state} Trials\{delay}\Female vs Male\{session.replace("No ","")}_{delay}_Ridge_{band}_{state}.pdf')
                


import extrapy.Scalogram as scalogram
import extrapy.Organize as og
import extrapy.Behaviour as B
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


savedir = r'D:\F.LARENO.FACCINI\RESULTS\New Results\LFP\Ridge'
mice = {
    '14': (6401, 6402, 6409),
    '15': (173, 176),
    '16': (6924,),
    '17': (6456, 6457)
}
females = (6409, 173, 176, 6456, 6457)
state = 'Good'
band = 'Delta'
####Power extraction####
power_mean_dic,tfr_sampling_rate = power_extraction(mice,females)


og.pickle_saving(savedir+fr'\Dataframe\{band}_{state}_Trials__tfrSR_{tfr_sampling_rate}',power_mean_dic)
# power_mean_dic = og.pickle_loading(savedir+r'\Dataframe\Delta_Good_Trials__tfrSR_16.0')
# tfr_sampling_rate = 16.0

plot_maker(power_mean_dic, band=band, state=state, tfr_sampling_rate=tfr_sampling_rate, save=True) 
plot_sex(power_mean_dic, females, band=band, state=state, tfr_sampling_rate=tfr_sampling_rate, save=True)


                
