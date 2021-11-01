# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 16:33:30 2021

@author: F.LARENO-FACCINI
"""
import sys
sys.path.append('D:\Pierre.LE-CABEC\Code Pierre')
import extrapy.Behaviour as bv
import extrapy.Organize as og
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import sklearn.metrics as metrics
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
 
def wheel_single_mouse(mice, protocols, females, delays,only_good=False):
    for delay in delays:
        if delay=='Fixed':
            skip_last=False
            return_delays=False
        else:
            skip_last=True
            return_delays=True
            
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
                    
                    good_trials,*(random,ot) = og.remove_empty_trials(lick_path,end_time='Reward',skip_last=skip_last,return_delays=return_delays)
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
                        plt.axvspan(0,0.15,alpha=0.2,color='r')

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

def wheel_avg_mice(mice, protocols, females, delays,only_good=False, shuffle=False, seed=np.random.randint(1,1000)):
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
                if group == '14' and protocol == 'P0':
                    continue
                else:
                    for mouse in topi:
                        i+=1
                        print(delay, group, mouse, protocol, i)
                                        
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
                        
                        if shuffle:
                            np.random.seed(seed)
                            try:
                                np.random.shuffle(ns[:,1])
                                np.random.shuffle(s[:,1])
                            except IndexError:
                                pass
                            
                            # Uncoment to center around the reward for Random Delay
                        # if delay=='Random': 
                        #     ns = center_to_zero(ns, random,ot)
                        #     s = center_to_zero(s, random,ot)
                        #     start = -3
                        #     stop = 7
    
                        # else:
                        start=0
                        stop=10

                        
                        if len(ns)>0:
                            new_ns = average_bin(ns,start=start,stop=stop)[2:,:]
                            if i == 1:
                                all_avg_ns = new_ns
                            else:
                                all_avg_ns = np.dstack((all_avg_ns,new_ns))


                        
                        if len(s)>0:
                            new_s = average_bin(s,start=start,stop=stop)[2:,:]
                            if i == 1:
                                all_avg_s = new_s
                            else:
                                all_avg_s = np.dstack((all_avg_s,new_s))

                    
                        print(all_avg_s.shape)
                
        ns_sem = stats.sem(all_avg_ns,axis=2)[:,1]
        s_sem = stats.sem(all_avg_s,axis=2)[:,1]


        all_avg_ns = np.mean(all_avg_ns, axis=2)
        all_avg_s = np.mean(all_avg_s, axis=2)
        
        ns_sem_convolved = gaussian_filter1d(ns_sem,sigma=1)
        s_sem_convolved = gaussian_filter1d(s_sem,sigma=1)
        
        ns_speed_convolved = gaussian_filter1d(all_avg_ns[:,1],sigma=1)
        s_speed_convolved = gaussian_filter1d(all_avg_s[:,1],sigma=1)
        
        statistic, pvalue = stats.ks_2samp(ns_speed_convolved, s_speed_convolved, alternative='two-sided')
        
        plt.figure()
        if delay == 'Random':
            plt.axvspan(0,0.5,alpha=0.2,color='k')
            plt.axvspan(1.5,2,alpha=0.2,color='k')
            plt.axvspan(2.45,2.60,alpha=0.2,color='r')
            plt.axvspan(2.95,3.10,alpha=0.2,color='r')
        else:
            plt.axvspan(0,0.5,alpha=0.2,color='k')
            plt.axvspan(1.5,2,alpha=0.2,color='k')
            plt.axvspan(2.55,2.7,alpha=0.2,color='r')
        
        
        plt.plot(all_avg_ns[:,0],ns_speed_convolved,label='Control')
        plt.fill_between(all_avg_ns[:,0], ns_speed_convolved+ns_sem_convolved,ns_speed_convolved-ns_sem_convolved,alpha=0.2)
        
        plt.plot(all_avg_s[:,0],s_speed_convolved, label='Photostim.')
        plt.fill_between(all_avg_s[:,0], s_speed_convolved+s_sem_convolved,s_speed_convolved-s_sem_convolved,alpha=0.2)
        
        plt.ylim(15,40)
        
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Average Instantaneous Speed (cm/s)')
        plt.title(f'Wheel speed {delay} Delay\nKolmogornov: {pvalue}')
        if shuffle:
            plt.savefig(fr'D:\Pierre.LE-CABEC\Results\Behaviour\Wheel\all mice\only good\shuffle\{delay}Delay.pdf')

        else:
            plt.savefig(fr'D:\Pierre.LE-CABEC\Results\Behaviour\Wheel\all mice\only good\{delay}Delay.pdf')
            
def lin_reg(wheel, reward_time=2.5, color='b', label='not available', samp_period = 0.1, shuffle=False, seed=np.random.randint(1,1000), display_fig=False):
        
    if len(wheel) >0:
        time_lr = int(reward_time/samp_period)
        # print('time',time_lr)
        
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(wheel[:time_lr,1:])
            
        slope, intercept, *_ = stats.linregress(wheel[:time_lr,0], wheel[:time_lr,1])
        fit = slope*(wheel[:time_lr,0]) + (intercept)
        
        if display_fig:
            plt.plot(wheel[:,0],wheel[:,1], color=color, label=label)
            plt.plot(wheel[:time_lr,0], fit, alpha=0.8, color=color)
        
        auc = metrics.auc(wheel[:time_lr,0], wheel[:time_lr,1])
        # print(len(bins[:time_lr]), len(n[:time_lr]))
        # print(slope,intercept,auc)
        
        return slope, intercept, auc
    else:
        return np.nan, np.nan, np.nan

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

def wheel_slope_mice(mice, protocols, females, only_good=False, shuffle=False, seed=np.random.randint(1,1000), display_fig=False):
    
    delays = ('Fixed', 'Random')
    list_ratios = []    
    for delay in delays:
        if delay=='Fixed':
            skip_last=False
            return_delays=False
        else:
            skip_last=True
            return_delays=True
        for protocol in protocols:
               
            
            for group,topi in mice.items():
                if group == '14' and protocol == 'P0':
                    continue
                else:
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
                        
                        if protocol != 'P0':
                            ns = v[(v[:,0] < 31)]
                            s = v[(v[:,0] > 30)]
                            if only_good:
                                ns = select_good_trials(ns, good_trials)
                                s = select_good_trials(s, good_trials)
                        else:
                            ns = v
                            if only_good:
                                ns = select_good_trials(ns, good_trials)
                          
                        start=0
                        stop=10
    
                        if delay == 'Fixed Delay':
                            reward_time = 2.5+(np.mean(ot)/1000)
                        else:
                            reward_time = 2.4+(np.mean(ot)/1000)
                        
                        if display_fig:
                            plt.figure()
                            
                        cm = ['#1f77b4','#ff7f0e']
                        
                        if len(ns)>0:
                            new_ns = average_bin(ns,start=start,stop=stop)[2:,:]
                            ns_slope, ns_intercept, ns_auc = lin_reg(new_ns, reward_time=reward_time, color=cm[0],label='Control', shuffle=shuffle, seed=seed, display_fig=display_fig)
                            ns_temp = [ns_slope,ns_intercept,ns_auc,mouse,sex,delay, 'No Stim', protocol] # build the dataframe
                            list_ratios.append(ns_temp)
                        else:
                            print('No Stim is empty\n')
                        
                        if protocol != 'P0':
                            if len(s)>0:
                                new_s = average_bin(s,start=start,stop=stop)[2:,:]
                                s_slope,s_intercept,s_auc = lin_reg(new_s, reward_time=reward_time, color=cm[1],label='Control',shuffle=shuffle, seed=seed, display_fig=display_fig)
                                s_temp = [s_slope,s_intercept,s_auc,mouse,sex,delay, 'Stim', protocol] # build the dataframe
                                list_ratios.append(s_temp)
                            else:
                                print('Stim is empty\n')
                            
                        if display_fig:
                            plt.title('{} {} {} [Control: {:.4f}, Stim: {:.4f}]'.format(mouse,protocol,delay,ns_slope,s_slope))
                            plt.xlabel('Time (s)')
                            plt.ylabel('Density of wheel speed')
                            plt.legend()

    return pd.DataFrame(list_ratios,columns = ('Slope', 'Intercept', 'AUC', 'Mouse', 'Sex', 'Delay', 'Condition', 'Protocol'))    


def plot_average_slope(list_ratios, mice, save=False, save_path=None, density=True, display_fig=True):
    
    slope_nostim_Fixed = list_ratios.loc[(list_ratios['Condition']=='No Stim')&(list_ratios['Delay']=='Fixed')][['Slope', 'Mouse']]
    slope_stim_Fixed = list_ratios.loc[(list_ratios['Condition']=='Stim')&(list_ratios['Delay']=='Fixed')][['Slope', 'Mouse']]
    
    slope_nostim_Random = list_ratios.loc[(list_ratios['Condition']=='No Stim')&(list_ratios['Delay']=='Random')][['Slope', 'Mouse']]
    slope_stim_Random = list_ratios.loc[(list_ratios['Condition']=='Stim')&(list_ratios['Delay']=='Random')][['Slope', 'Mouse']]
    
    slope_dic_temp = {'Fixed Delay': {'nostim': slope_nostim_Fixed, 'stim': slope_stim_Fixed}, 'Random Delay': {'nostim': slope_nostim_Random, 'stim': slope_stim_Random}}
    
    slope_dic ={}
    
    for delay, condition_dic in slope_dic_temp.items():
        slope_dic[delay] = {'nostim' : {'mouse':[], 'sem':[], 'mean':[]}, 'stim': {'mouse':[], 'sem':[], 'mean':[]}}
       
        
        for group, list_mouse in mice.items():
            for mouse in list_mouse:
                print(mouse)
                slope_dic[delay]['nostim']['mouse'].append(mouse)
                slope_dic[delay]['stim']['mouse'].append(mouse)
                
                slope_dic[delay]['nostim']['sem'].append(stats.sem(condition_dic['nostim'].loc[(condition_dic['nostim']['Mouse']==mouse)]['Slope']))
                slope_dic[delay]['stim']['sem'].append(stats.sem(condition_dic['stim'].loc[(condition_dic['stim']['Mouse']==mouse)]['Slope']))
                
                slope_dic[delay]['nostim']['mean'].append(np.mean(condition_dic['nostim'].loc[(condition_dic['nostim']['Mouse']==mouse)]['Slope']))
                slope_dic[delay]['stim']['mean'].append(np.mean(condition_dic['stim'].loc[(condition_dic['stim']['Mouse']==mouse)]['Slope']))
    

        slope_dic[delay]['nostim'] = pd.DataFrame(slope_dic[delay]['nostim'])
        slope_dic[delay]['stim'] = pd.DataFrame(slope_dic[delay]['stim'])
        
         
        if display_fig:
            fig, ax = plt.subplots(figsize=(8,8))
            cm = ['#1f77b4','#ff7f0e']
            
            if density:
                _, bins_nostim, _ = ax.hist(slope_dic[delay]['nostim']['mean'], alpha=0.5, density=True, color=cm[0], label='No Stim')
                sigma_nostim = np.std(slope_dic[delay]['nostim']['mean'])
                mu_nostim = np.mean(slope_dic[delay]['nostim']['mean'])  
                y_nostim = ((1 / (np.sqrt(2 * np.pi) * sigma_nostim)) *np.exp(-0.5 * (1 / sigma_nostim * (bins_nostim - mu_nostim))**2))
                ax.plot(bins_nostim, y_nostim, color=cm[0]) 
                ax.axvline(mu_nostim, color=cm[0], ls='--')
               
                _, bins_stim, _ = ax.hist(slope_dic[delay]['stim']['mean'], alpha=0.5, density=True, color=cm[1], label='Stim')
                sigma_stim = np.std(slope_dic[delay]['stim']['mean'])
                mu_stim = np.mean(slope_dic[delay]['stim']['mean'])
                y_stim = ((1 / (np.sqrt(2 * np.pi) * sigma_stim)) *np.exp(-0.5 * (1 / sigma_stim * (bins_stim - mu_stim))**2))
                ax.plot(bins_stim, y_stim, color=cm[1])
                ax.axvline(mu_stim, color=cm[1], ls='--')
                
                effectsize = round(abs(mu_stim-mu_nostim)/np.mean([sigma_stim,sigma_nostim]), 3)
                statistic, pvalue = stats.ks_2samp(slope_dic[delay]['stim']['mean'], slope_dic[delay]['nostim']['mean'], alternative='two-sided')
                
                ax.set_title(f'Distribution of the mean slope of the wheel speed (pre reward) per animal.\n{delay}\nKolomogornov: {pvalue}\nEffect size: {effectsize}')
                ax.set_xlabel('slope')
                ax.set_ylabel('probability density')
                ax.legend()
                if save:
                    plt.savefig(fr'{save_path}\ProbDens_WheelSlope_{delay}_StimVsNoStim.pdf')
            
            else:
                ax.hist(slope_dic[delay]['nostim']['mean'], alpha=0.5, density=False, color=cm[0], label='No Stim')
               
                ax.hist(slope_dic[delay]['stim']['mean'], alpha=0.5, density=False, color=cm[1], label='Stim')
                
                
                ax.set_title(f'Distribution of the mean slope of the wheel speed (pre reward) per animal.\n{delay}')
                ax.set_xlabel('slope')
                ax.set_ylabel('Count nb')
                ax.legend()
                if save:
                    plt.savefig(fr'{save_path}\WheelSlope_{delay}_StimVsNoStim.pdf')  
    
    return slope_dic


def stats_stimvsnostim(slope_dic, savedir, save=False):
    stat_dic = {}
    stat_to_save = ['Wheel speed slope from 0 to 2.5s (fixed) or 2.4s (random), Stim Vs No Stim, Paired T-test\n\n']
    for delay, delay_dic in slope_dic.items():
        delay_dic['nostim'].sort_values(by=['mouse'], inplace=True)
        delay_dic['stim'].sort_values(by=['mouse'], inplace=True)
        W,p_value = stats.ttest_rel(delay_dic['nostim']['mean'],delay_dic['stim']['mean'],alternative='two-sided')
        stat_dic[delay] = {'Statistic': W, 'p_value': p_value}
        stat_to_save.append(f'{delay}:\n')
        stat_to_save.append(f'     Statistic: {W}\n')
        stat_to_save.append(f'     P_value: {p_value}\n\n')
    
    if save:
        with open(f'{save_path}\Stats_WheelSlope_StimvsNoStim.txt', 'w') as stats_file:
                stats_file.writelines(stat_to_save)
            
    return stat_dic

def stats_shufflevsnotshuffle(slope_dic, slope_shuffle_dic, save_path, save=False, seed=np.random.randint(1,1000)):
    stat_dic = {}
    stat_to_save = [f'Wheel speed slope from 0 to 2.5s (fixed) or 2.4s (random), shuffle Vs not shuffle (seed={seed}), two-sample Kolmogorov-Smirnov test\n\n']

    for delay, delay_dic in slope_dic.items():
        stat_to_save.append(f'{delay}:\n\n')
        for condition, df in delay_dic.items():
            statistic, pvalue = stats.ks_2samp(df['mean'], slope_shuffle_dic[delay][condition]['mean'], alternative='two-sided')
            stat_dic[delay] = {'Statistic': statistic, 'p_value': pvalue}
            
            stat_to_save.append(f'{condition}:\n')
            stat_to_save.append(f'     Statistic: {statistic}\n')
            stat_to_save.append(f'     P_value: {pvalue}\n\n')
    
    if save:
        with open(f'{save_path}\Stats_WheelSlope_StimvsNoStim.txt', 'w') as stats_file:
                stats_file.writelines(stat_to_save)

    return stat_dic

def MAD(a,axis=None):
     '''
     Computes median absolute deviation of an array along given axis
     '''
     #Median along given axis but keep reduced axis so that result can still broadcast along a

     med = np.nanmedian(a, axis=axis, keepdims=True)
     mad = np.nanmedian(np.abs(a-med),axis=axis) #MAD along the given axis

     return mad
 
def bootstrap_patterns(slope_dic, run=1000, N=11, input_method='average', output_method='average'):
    
    '''
    Bootstraps synaptic patterns and returns median or average pattern
    
    patterns (list of arrays) : the patterns (data)
    run (int) : the amount of runs for average/median
    N (int) : number of draws for each cycle
    input_method (str) : 'median' , 'average' stores median or average value for each run 
    output_method (str) : 'median' or 'average' : returns medianed or averaged pattern
    
    '''
    boot_dic = {}
    for delay, delay_dic in slope_dic.items():
        boot_dic[delay] = {}
        for condition, df in delay_dic.items():
            patterns = df['mean']
            endCycle = []
            
            for i in range(run): 
                
                temp = []
                
                for j in range(N):
                
                    randIndex = np.random.randint(0,len(patterns),size=1)[0]
                                
                    temp.append(patterns[randIndex])
                    
                    if len(temp) == N:
                        pass
                    else:
                        continue
                        
                if input_method == 'median' : 
                    endCycle.append(np.nanmedian(temp, axis=0))
                
                elif input_method == 'average' : 
                    endCycle.append(np.nanmean(temp, axis=0))
        
        
            if output_method == 'median': 
                out_bootstrap = np.nanmedian(endCycle, axis=0) 
                out_deviation = MAD(endCycle, axis=0)
                
            elif output_method == 'average': 
                out_bootstrap = np.nanmean(endCycle, axis=0)
                out_deviation = np.nanstd(endCycle, axis=0)
                
            boot_dic[delay][condition] = {'out_bootstrap': np.asarray(out_bootstrap), 'out_deviation': np.asarray(out_deviation), 'endCycle': endCycle}
        
    return boot_dic 

def bootstrap_patterns_shuffle(mice,protocols,females, run=1000, N=11, input_method='average', output_method='average'):
    
    '''
    Bootstraps synaptic patterns and returns median or average pattern
    
    patterns (list of arrays) : the patterns (data)
    run (int) : the amount of runs for average/median
    N (int) : number of draws for each cycle
    input_method (str) : 'median' , 'average' stores median or average value for each run 
    output_method (str) : 'median' or 'average' : returns medianed or averaged pattern
    
    '''
    
    boot_dic = {'Fixed Delay': {'nostim': {'endCycle': []}, 'stim': {'endCycle': []}}, 'Random Delay': {'nostim': {'endCycle': []}, 'stim': {'endCycle': []}}}     
    for i in range(run): 
        seed=np.random.randint(1,100000)
        list_ratios_shuffle = wheel_slope_mice(mice,protocols,females,only_good=True, shuffle=True, seed=seed, display_fig=False)
        slope_shuffle_dic = plot_average_slope(list_ratios_shuffle, mice, save=False, density=True, display_fig=False)

        for delay, delay_dic in slope_shuffle_dic.items():
            for condition, df in delay_dic.items():
                patterns = df['mean'] 
                temp = []
                
                for j in range(N):
                
                    randIndex = np.random.randint(0,len(patterns),size=1)[0]
                                
                    temp.append(patterns[randIndex])
                    
                    if len(temp) == N:
                        pass
                    else:
                        continue
                        
                if input_method == 'median' : 
                    boot_dic[delay][condition]['endCycle'].append(np.nanmedian(temp, axis=0))
                
                elif input_method == 'average' : 
                    boot_dic[delay][condition]['endCycle'].append(np.nanmean(temp, axis=0))

    for delay, delay_dic in boot_dic.items():
        for condition, condition_dic in delay_dic.items():
            if output_method == 'median': 
                out_bootstrap = np.nanmedian(condition_dic['endCycle'], axis=0) 
                out_deviation = MAD(condition_dic['endCycle'], axis=0)
                
            elif output_method == 'average': 
                out_bootstrap = np.nanmean(condition_dic['endCycle'], axis=0)
                out_deviation = np.nanstd(condition_dic['endCycle'], axis=0)
                
            boot_dic[delay][condition]['out_bootstrap']= np.asarray(out_bootstrap)
            boot_dic[delay][condition]['out_deviation']= np.asarray(out_deviation)

    return boot_dic 

def plot_boostrap(boot_dic, boot_dic_shuffle, density=True, save=False, save_path=None):
    cm = ['#1f77b4','#ff7f0e']
    for delay, delay_dic in boot_dic.items():
        for condition, condition_dic in delay_dic.items():
            fig, ax = plt.subplots(figsize=(8,8))
            if not density:
                ax.hist(condition_dic['endCycle'], color=cm[0], label='not Shuffle', alpha=0.5)
                ax.hist(boot_dic_shuffle[delay][condition]['endCycle'], color=cm[1], label='Shuffle', alpha=0.5)
                statistic, pvalue = stats.ks_2samp(condition_dic['endCycle'], boot_dic_shuffle[delay][condition]['endCycle'], alternative='two-sided')
                
                sigma_notshuffle = np.std(condition_dic['endCycle'])
                mu_notshuffle = np.mean(condition_dic['endCycle'])  
                sigma_shuffle = np.std(boot_dic_shuffle[delay][condition]['endCycle'])
                mu_shuffle = np.mean(boot_dic_shuffle[delay][condition]['endCycle'])
                effectsize = round(abs(mu_shuffle-mu_notshuffle)/np.mean([sigma_notshuffle,sigma_shuffle]), 3)
                ax.set_title(f'Bootstrap wheel speed slope distribution, shuffle vs not shuffle,\n{delay} {condition},\np_value={pvalue}(two-sample Kolmogorov-Smirnov test)\nEffect size: {effectsize}')
                ax.legend()
                if save:
                    plt.savefig(save_path+fr'\bootstrap\100Cycle\\{delay}_{condition}_ShuffleVSNotShuffle.pdf')
                
    
            else:
                _, bins_notshuffle, _ = ax.hist(condition_dic['endCycle'], alpha=0.5, density=True, color=cm[0], label='Not shuffle')
                sigma_notshuffle = np.std(condition_dic['endCycle'])
                mu_notshuffle = np.mean(condition_dic['endCycle'])  
                y_notshuffle = ((1 / (np.sqrt(2 * np.pi) * sigma_notshuffle)) *np.exp(-0.5 * (1 / sigma_notshuffle * (bins_notshuffle - mu_notshuffle))**2))
                ax.plot(bins_notshuffle, y_notshuffle, color=cm[0]) 
                ax.axvline(mu_notshuffle, color=cm[0], ls='--')
               
                _, bins_shuffle, _ = ax.hist(boot_dic_shuffle[delay][condition]['endCycle'], alpha=0.5, density=True, color=cm[1], label='shuffle')
                sigma_shuffle = np.std(boot_dic_shuffle[delay][condition]['endCycle'])
                mu_shuffle = np.mean(boot_dic_shuffle[delay][condition]['endCycle'])
                y_shuffle = ((1 / (np.sqrt(2 * np.pi) * sigma_shuffle)) *np.exp(-0.5 * (1 / sigma_shuffle * (bins_shuffle - mu_shuffle))**2))
                ax.plot(bins_shuffle, y_shuffle, color=cm[1])
                ax.axvline(mu_shuffle, color=cm[1], ls='--')
                
                statistic, pvalue = stats.ks_2samp(condition_dic['endCycle'], boot_dic_shuffle[delay][condition]['endCycle'], alternative='two-sided')
                effectsize = round(abs(mu_shuffle-mu_notshuffle)/np.mean([sigma_notshuffle,sigma_shuffle]), 3)
                
                ax.set_title(f'Bootstrap wheel speed slope distribution, shuffle vs not shuffle,\n{delay} {condition},\np_value={pvalue}(two-sample Kolmogorov-Smirnov test)\nEffect size: {effectsize}')
                ax.legend()
                if save:
                    plt.savefig(save_path+fr'\bootstrap\100Cycle\\\density_{delay}_{condition}_ShuffleVSNotShuffle.pdf')

def plot_boostrap_fixe_vs_random(boot_dic, density=True, savedir=None):
    cm = ['#1f77b4','#ff7f0e']
    for condition in ['stim', 'nostim']:
        fig, ax = plt.subplots(figsize=(8,8))
        
        fixe_data = boot_dic['Fixed Delay'][condition]['endCycle']
        variable_data = boot_dic['Random Delay'][condition]['endCycle']
        if not density:
            ax.hist(fixe_data, color=cm[0], label='Fixed Delay', alpha=0.5)
            ax.hist(variable_data, color=cm[1], label='Variable Delay', alpha=0.5)
            
            statistic, pvalue = stats.ks_2samp(fixe_data, variable_data, alternative='two-sided')
            
            sigma_fixe = np.std(fixe_data)
            mu_fixe = np.mean(fixe_data)  
            sigma_variable = np.std(variable_data)
            mu_variable = np.mean(variable_data)
            effectsize = round(abs(mu_variable-mu_fixe)/np.mean([sigma_fixe,sigma_variable]), 3)
            ax.set_title(f'Bootstrap lick slope distribution, Fixed vs not Variable,\n{condition}, \np_value={pvalue}(two-sample Kolmogorov-Smirnov test)\nEffect size: {effectsize}')
            ax.legend()
            if savedir is not None:
                plt.savefig(savedir+fr'\bootstrap\100Cycle\Fixe vs Variable\\{condition}_FixedVSVariable.pdf')

        else:
            n, bins_fixe, _ = ax.hist(fixe_data, alpha=0.5, density=True, color=cm[0], label='Fixed Delay')
            sigma_fixe = np.std(fixe_data)
            mu_fixe = np.mean(fixe_data)  
            y_fixe = ((1 / (np.sqrt(2 * np.pi) * sigma_fixe)) *np.exp(-0.5 * (1 / sigma_fixe * (bins_fixe - mu_fixe))**2))
            ax.plot(bins_fixe, y_fixe, color=cm[0]) 
            ax.axvline(mu_fixe, color=cm[0], ls='--')
           
            _, bins_variable, _ = ax.hist(variable_data, alpha=0.5, density=True, color=cm[1], label='Variable Delay')
            sigma_variable = np.std(variable_data)
            mu_variable = np.mean(variable_data)
            y_variable = ((1 / (np.sqrt(2 * np.pi) * sigma_variable)) *np.exp(-0.5 * (1 / sigma_variable * (bins_variable - mu_variable))**2))
            ax.plot(bins_variable, y_variable, color=cm[1])
            ax.axvline(mu_variable, color=cm[1], ls='--')
            
            statistic, pvalue = stats.ks_2samp(fixe_data, variable_data, alternative='two-sided')
            effectsize = round(abs(mu_variable-mu_fixe)/np.mean([sigma_fixe,sigma_variable]), 3)
            ax.set_title(f'Bootstrap lick slope distribution, Fixed vs not Variable,\n{condition}, \np_value={pvalue}(two-sample Kolmogorov-Smirnov test)\nEffect size: {effectsize}')
            ax.legend()
            
            if savedir is not None:
                plt.savefig(savedir+fr'\bootstrap\100Cycle\Fixe vs Variable\\density_{condition}_FixedVSRandom.pdf')


if __name__ == '__main__':

    save_path = r'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\RESULTS\Behaviour\Wheel\LinReg'
    # protocols = ('P0','P13', 'P15' ,'P16', 'P18','Washout')
    protocols = ('P0', 'P13', 'P15' ,'P16', 'P18')

    mice = {'14': (6401, 6402, 6409),
            '15': (173, 174, 176),
            '16': (6924, 6928, 6934),
            '17': (6456, 6457)}
    
    females = (6409, 173, 174, 176, 6934, 6456, 6457)
    delays = ('Fixed', 'Random')
    seed = 876
    
    # wheel_single_mouse(mice,protocols,females,delays,only_good=True)
    # wheel_avg_mice(mice,protocols,females,delays,only_good=True, shuffle=False, seed=seed)
    
    # list_ratios = wheel_slope_mice(mice,protocols,females,only_good=True, display_fig=False)
    # og.pickle_saving(r'\\equipe2-nas2\Pierre.LE-CABEC\Results\Behaviour\Wheel\mean slop by animal\\wheel_df', list_ratios)
    # list_ratios = og.pickle_loading(r'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\RESULTS\Behaviour\Wheel\LinReg\\wheel_df')
      
    # slope_dic = plot_average_slope(list_ratios, mice, save=True, save_path=save_path, density=True, display_fig=True)
    # og.pickle_saving(r'\\equipe2-nas2\Pierre.LE-CABEC\Results\Behaviour\Wheel\mean slop by animal\\slope_dic', slope_dic)
    # # slope_dic = og.pickle_loading(r'\\equipe2-nas2\Pierre.LE-CABEC\Results\Behaviour\Wheel\mean slop by animal\\slope_dic')
    # # stats_dic = stats_stimvsnostim(slope_dic, save_path)
         
    # boot_dic = bootstrap_patterns(slope_dic, run=100, N=11)
    # og.pickle_saving(r'\\equipe2-nas2\Pierre.LE-CABEC\Results\Behaviour\Wheel\mean slop by animal\bootstrap\100Cycle\\boot_dic', boot_dic)
    boot_dic = og.pickle_loading(r'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\RESULTS\Behaviour\Wheel\LinReg\bootstrap\100Cycle\\boot_dic')


    # boot_dic_shuffle = bootstrap_patterns_shuffle(mice,protocols,females, run=100, N=11)
    # og.pickle_saving(r'\\equipe2-nas2\Pierre.LE-CABEC\Results\Behaviour\Wheel\mean slop by animal\bootstrap\100Cycle\\boot_dic_shuffle', boot_dic_shuffle)
    boot_dic_shuffle = og.pickle_loading(r'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\RESULTS\Behaviour\Wheel\LinReg\bootstrap\100Cycle\\boot_dic_shuffle')

    
    # plot_boostrap(boot_dic, boot_dic_shuffle, density=True, save=False, save_path=save_path)
    plot_boostrap_fixe_vs_random(boot_dic, density=True, savedir=save_path)

'''
    path = r'//equipe2-nas2/F.LARENO-FACCINI/BACKUP FEDE/Behaviour/Group 14/6401 (CM16-Buz - Male)/Random Delay/P13/6401_2020_06_11_14_28_36.coder'
    
    wheel = bv.load_lickfile(path,wheel=True)
    v = np.array([[t,w, (3.875/(w-wheel[indx-1,1]))]for indx,(t,w) in enumerate(wheel)]) # cm/s
    
    lick_path = path.replace('.coder','.lick')
    good_trials,*(random,ot) = og.remove_empty_trials(lick_path,end_time='Reward',skip_last=True,return_delays=True)
    good_trials = np.unique(good_trials[:,0])
    random = np.array(random)
    
    ns = v[(v[:,0] < 31)]
    ns = select_good_trials(ns, good_trials)
    ns = center_to_zero(ns, random,ot)
    
    s = v[(v[:,0] > 30)]
    s = select_good_trials(s, good_trials)
    s = center_to_zero(s, random)
    
    new_ns = average_bin(ns,start=-3,stop=7)
    new_s = average_bin(s,start=-3,stop=7)
    

    plt.figure()
    # plt.axvspan(0,0.5,alpha=0.2,color='k')
    # plt.axvspan(1.5,2,alpha=0.2,color='k')
    plt.axvspan(0,0.5,alpha=0.2,color='r')
    
    
    plt.plot(new_ns[:,0],new_ns[:,1],label='Control')
    plt.fill_between(new_ns[:,0], new_ns[:,1]+new_ns[:,2],new_ns[:,1]-new_ns[:,2],alpha=0.2)
    
    plt.plot(new_s[:,0],new_s[:,1], label='Photostim.')
    plt.fill_between(new_s[:,0], new_s[:,1]+new_s[:,2],new_s[:,1]-new_s[:,2],alpha=0.2)
    
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Average Instantaneous Speed (cm/s)')
    plt.title('Wheel speed 6409 P18')
'''