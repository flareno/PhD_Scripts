# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 16:33:30 2021

@author: F.LARENO-FACCINI


Computes instantaneous speed of the mouse using positional data from the wheel
Averages it over the session for each mouse and then plots the average of all the mice

"""
import extrapy.Behaviour as bv
import extrapy.Organize as og
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd

def average_bin(array,samp_period=0.1,start=0,stop=10):
    '''
    Returns
    -------
    new_wheel : 2D NumPy array
        [:,0] bins by time (defined by the sampling period)
        [:,1] mean of that bin across the session
        [:,2] std of that bin across the session
    '''
    
    bins = np.arange(start,stop,samp_period)
    
    new_wheel = []
    for ind,i in enumerate(bins):
        if ind>0:
            intervallo = (bins[ind-1],i)
            temp__ = array[(array[:,1] > intervallo[0]) & (array[:,1] < intervallo[1])]
            new_wheel.append((i,np.mean(temp__[:,2]),np.std(temp__[:,2])))
   
    new_wheel = np.array(new_wheel)
    new_wheel = np.nan_to_num(new_wheel)

    return new_wheel

def select_good_trials(array, good_trials):
    prova = []
    for x in array:
        if x[0] in good_trials:
            prova.append(x)
    return np.array(prova)


def center_to_zero(array,random,ot):
    random = np.array(random)
    for x in array:
        curr_delay = (random[random[:,1] == x[0]][0][0])/1000
        time_reward = 2+curr_delay+ot
        # print(x[0], time_reward)
        x[1] = x[1]-time_reward
    return array


def wheel_avg_mice(mice, protocols, females, delays,only_good=False):
    for delay in delays:
        if delay=='Fixed':
            skip_last=False
            return_delays=False
        else:
            skip_last=True
            return_delays=True
        i=0
        for protocol in protocols:
            
            
            for group,topi in mice.items():
                
                for mouse in topi:
                    i+=1
                    print(delay, group, mouse, protocol, i)
                                    
                    sex = 'Female' if mouse in females else 'Male'
                    
                    path = fr'//equipe2-nas2/F.LARENO-FACCINI/BACKUP FEDE/Behaviour/Group {group}/{mouse} (CM16-Buz - {sex})/{delay} Delay/{protocol}'
                    file = path+'/'+og.file_list(path,False, ext='.coder')[0]
                    
                    wheel = bv.load_lickfile(file,wheel=True)
                    v = np.array([[t,w, (3.875/(w-wheel[indx-1,1]))]for indx,(t,w) in enumerate(wheel)]) # cm/s
                    
                    lick_path = file.replace('.coder','.lick')
                    good_trials,*(random,ot) = og.remove_empty_trials(lick_path,end_time='Reward',skip_last=skip_last,return_delays=return_delays)
                    good_trials = np.unique(good_trials[:,0])
                    
                    ns = v[(v[:,0] < 31)]
                    s = v[(v[:,0] > 30)]
                    if only_good:
                        ns = select_good_trials(ns, good_trials)
                        s = select_good_trials(s, good_trials)
                        
                    if delay=='Random':
                        ns = center_to_zero(ns, random,ot)
                        s = center_to_zero(s, random,ot)
                        start = -3
                        stop = 7

                    else:
                        start=0
                        stop=10

                        
                    if len(ns)>0:
                        new_ns = average_bin(ns,start=start,stop=stop)
                        if i == 1:
                            all_avg_ns = new_ns
                        else:
                            all_avg_ns = np.dstack((all_avg_ns,new_ns))
                    
                    if len(s)>0:
                        new_s = average_bin(s,start=start,stop=stop)
                        if i == 1:
                            all_avg_s = new_s
                        else:
                            all_avg_s = np.dstack((all_avg_s,new_s))
                
                    print(all_avg_s.shape)
                
        ns_sem = stats.sem(all_avg_ns,axis=2)[:,1]
        s_sem = stats.sem(all_avg_s,axis=2)[:,1]


        all_avg_ns = np.mean(all_avg_ns, axis=2)
        all_avg_s = np.mean(all_avg_s, axis=2)
        
        plt.figure()
        if delay == 'Random':
            # plt.axvspan(0,0.5,alpha=0.2,color='k')
            # plt.axvspan(1.5,2,alpha=0.2,color='k')
            plt.axvspan(0,0.15,alpha=0.2,color='r')
        else:
            plt.axvspan(0,0.5,alpha=0.2,color='k')
            plt.axvspan(1.5,2,alpha=0.2,color='k')
            plt.axvspan(2.55,2.7,alpha=0.2,color='r')
        
        
        plt.plot(all_avg_ns[:,0],all_avg_ns[:,1],label='Control')
        plt.fill_between(all_avg_ns[:,0], all_avg_ns[:,1]+ns_sem,all_avg_ns[:,1]-ns_sem,alpha=0.2)
        
        plt.plot(all_avg_s[:,0],all_avg_s[:,1], label='Photostim.')
        plt.fill_between(all_avg_s[:,0], all_avg_s[:,1]+s_sem,all_avg_s[:,1]-s_sem,alpha=0.2)
        
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Average Instantaneous Speed (cm/s)')
        plt.title(f'Wheel speed {delay} Delay')
        # plt.savefig(fr'D:\F.LARENO.FACCINI\RESULTS\Behaviour\Wheel\all mice\{delay}Delay.pdf')

    
def wheel_single_mouse(mice, protocols, females, delays,only_good=False):
    for delay in delays:
        skip_last=False if delay == 'Fixed' else True
            
        for protocol in protocols:
            
            for group,topi in mice.items():
                
                for mouse in topi:
                    print(delay, group, mouse, protocol)
                                    
                    sex = 'Female' if mouse in females else 'Male'
                    
                    path = fr'//equipe2-nas2/F.LARENO-FACCINI/BACKUP FEDE/Behaviour/Group {group}/{mouse} (CM16-Buz - {sex})/{delay} Delay/{protocol}'
                    file = path+'/'+og.file_list(path,False, ext='.coder')[0]
                    
                    wheel = bv.load_lickfile(file,wheel=True)
                    v = np.array([[t,w, (3.875/(w-wheel[indx-1,1]))]for indx,(t,w) in enumerate(wheel)]) # cm/s
                    
                    lick_path = file.replace('.coder','.lick')
                    
                    good_trials,*(random,ot) = og.remove_empty_trials(lick_path,end_time='Reward',skip_last=skip_last)
                    good_trials = np.unique(good_trials[:,0])
                                        
                    ns = v[(v[:,0] < 31)]
                    s = v[(v[:,0] > 30)]
                    if only_good:
                        ns = select_good_trials(ns, good_trials)
                        s = select_good_trials(s, good_trials)
                        
                    plt.figure()
                    
                    if delay=='Random':
                        ns = center_to_zero(ns, random,ot)
                        s = center_to_zero(s, random,ot)
                        start = -3
                        stop = 7
                        # plt.axvspan(0,0.5,alpha=0.2,color='k')
                        # plt.axvspan(1.5,2,alpha=0.2,color='k')
                        # plt.axvspan(0,0.15,alpha=0.2,color='r')

                    else:
                        start=0
                        stop=10
                        plt.axvspan(0,0.5,alpha=0.2,color='k')
                        plt.axvspan(1.5,2,alpha=0.2,color='k')
                        plt.axvspan(2.55,2.7,alpha=0.2,color='r')

                    
                    
                    
                    if len (ns) > 0:
                        new_ns = average_bin(ns,start=start,stop=stop)
                        plt.plot(new_ns[:,0],new_ns[:,1],label='Control')
                        plt.fill_between(new_ns[:,0], new_ns[:,1]+new_ns[:,2],new_ns[:,1]-new_ns[:,2],alpha=0.2)
                    
                    if len(s) > 0 :                     
                        new_s = average_bin(s,start=start,stop=stop)
                        plt.plot(new_s[:,0],new_s[:,1], label='Photostim.')
                        plt.fill_between(new_s[:,0], new_s[:,1]+new_s[:,2],new_s[:,1]-new_s[:,2],alpha=0.2)
                    
                    plt.legend()
                    plt.xlabel('Time (s)')
                    plt.ylabel('Average Instantaneous Speed (cm/s)')
                    plt.title(f'Wheel speed {mouse} {protocol} {delay} Delay')





def make_df_speed(mice, protocols, females, delays,only_good=False, plot=False):
    df_avg = None
    for delay in delays:
        skip_last=False if delay == 'Fixed' else True
          
        for protocol in protocols:
            
            for group,topi in mice.items():
                
                for mouse in topi:
                    print(delay, group, mouse, protocol)
                                    
                    sex = 'Female' if mouse in females else 'Male'
                    
                    path = fr'//equipe2-nas2/F.LARENO-FACCINI/BACKUP FEDE/Behaviour/Group {group}/{mouse} (CM16-Buz - {sex})/{delay} Delay/{protocol}'
                    file = path+'/'+og.file_list(path,False, ext='.coder')[0]
                    
                    wheel = bv.load_lickfile(file,wheel=True)
                    v = np.array([[t,w, (3.875/(w-wheel[indx-1,1]))]for indx,(t,w) in enumerate(wheel)]) # cm/s
                    
                    lick_path = file.replace('.coder','.lick')
                    
                    good_trials,*(random,ot) = og.remove_empty_trials(lick_path,end_time='Reward',skip_last=skip_last)
                    good_trials = np.unique(good_trials[:,0])
                                        
                    ns = v[(v[:,0] < 31)]
                    s = v[(v[:,0] > 30)]
                    if only_good:
                        ns = select_good_trials(ns, good_trials)
                        s = select_good_trials(s, good_trials)
                            
                    
                    if len (ns) > 0:
                        new_ns = average_bin(ns,start=0,stop=10)
                    
                    if len(s) > 0 :                     
                        new_s = average_bin(s,start=0,stop=10)


                    time = 2.5 if delay == 'Fixed' else 2.4
                    print(len(new_ns))
                    stop_ind = np.where(np.isclose(new_ns[:,0], time))[0][0]+1
                    mean_speed_ns = np.mean(new_ns[:stop_ind,1])
                    mean_speed_s = np.mean(new_s[:stop_ind,1])


                    if df_avg is None:
                        df_avg = pd.DataFrame({'Speed':[mean_speed_ns,mean_speed_s],
                                               'Mouse': [mouse, mouse],
                                               'Delay': [delay, delay],
                                               'Protocol':[protocol, protocol],
                                               'Condition': ['No Stim', 'Stim']}, index=None)
                    else:
                        temp__ = pd.DataFrame({'Speed':[mean_speed_ns,mean_speed_s],
                                               'Mouse': [mouse, mouse],
                                               'Delay': [delay, delay],
                                               'Protocol':[protocol, protocol],
                                               'Condition': ['No Stim', 'Stim']}, index=None)
                        df_avg = pd.concat((df_avg,temp__), axis=0)
    
    
    new_df_avg = None
    for _, values in mice.items():
        for delay in delays:
            for mouse in values:
                ns_temp = np.mean(df_avg[(df_avg.Mouse == mouse) & (df_avg.Condition == 'No Stim') & (df_avg.Delay == delay)].Speed)
                s_temp = np.mean(df_avg[(df_avg.Mouse == mouse) & (df_avg.Condition == 'Stim') & (df_avg.Delay == delay)].Speed)
    
                if new_df_avg is None:
                    new_df_avg = pd.DataFrame({'Speed':[ns_temp,s_temp],
                                           'Mouse': [mouse, mouse],
                                           'Delay': [delay, delay],
                                           'Condition': ['No Stim', 'Stim']}, index=None)
                else:
                    temp__ = pd.DataFrame({'Speed':[ns_temp,s_temp],
                                           'Mouse': [mouse, mouse],
                                           'Delay': [delay, delay],
                                           'Condition': ['No Stim', 'Stim']}, index=None)
                    new_df_avg = pd.concat((new_df_avg,temp__), axis=0)

    return new_df_avg






if __name__ == '__main__':
    import seaborn as sns

    protocols = ('P13', 'P15' ,'P16', 'P18')

    mice = {'14': (6401, 6402, 6409),
            '15': (173, 174, 176),
            '16': (6924, 6928, 6934),
            '17': (6456, 6457)}
    
    females = (6409, 173, 174, 176, 6934, 6456, 6457)
    delays = ('Fixed', 'Random')
    
    # wheel_single_mouse(mice,protocols,females,delays,only_good=True)
    # wheel_avg_mice(mice,protocols,females,delays,only_good=True)
    
    df_speed = make_df_speed(mice, protocols, females, delays, only_good=True)
    
    
    
    for cond in ('No Stim', 'Stim'):
        temp_fixed = df_speed[(df_speed.Condition == cond) & (df_speed.Delay == 'Fixed')]
        temp_random = df_speed[(df_speed.Condition == cond) & (df_speed.Delay == 'Random')]
        
        print(cond)
        print(np.mean(temp_fixed.Speed), stats.sem(temp_fixed.Speed))
        print(np.mean(temp_random.Speed), stats.sem(temp_random.Speed))
        
        # print(stats.shapiro(temp_fixed.Speed))
        # print(stats.shapiro(temp_random.Speed))
        
        _,pval = stats.ttest_rel(temp_fixed.Speed, temp_random.Speed)
        effect = og.effectsize(temp_fixed.Speed, temp_random.Speed)
        print(effect)
        # temp__ = pd.concat((temp_fixed, temp_random))
        # plt.figure(figsize=(15,7))
        # sns.boxplot(x=temp__['Delay'], y=temp__['Speed'])
        # plt.title(f'speed foreperiod during {cond}. p-value (paired t-test) {pval}')

    
