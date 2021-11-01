# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 13:23:30 2021

@author: F.LARENO-FACCINI
"""
import sys    
sys.path.append(r'\\equipe2-nas2\Pierre.LE-CABEC\Code Pierre')
import extrapy.Behaviour as bv
import extrapy.Organize as og
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
    
def lin_reg(licks, label, color, reward_time=2.5, samp_period = 0.05,shuffle=False, seed=100, display_fig=False):
    if len(licks) >0:
        time_lr = int(reward_time/samp_period)
        # print('time',time_lr)
        n,bins,*_ = bv.psth_lick(licks,samp_period=samp_period,density=True, label=label, color=color)
        if not display_fig:
            plt.close()
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(n[:time_lr])
        slope, intercept, *_ = stats.linregress(bins[:time_lr], n[:time_lr])
        fit = slope*(bins[:time_lr]) + (intercept)
        if display_fig:
            plt.plot(bins[:time_lr], fit, alpha=0.8, color=color)
        
        auc = metrics.auc(bins[:time_lr], n[:time_lr])
        # print(len(bins[:time_lr]), len(n[:time_lr]))
        # print(slope,intercept,auc)
        
        return slope, intercept, auc
    else:
        return np.nan, np.nan, np.nan

def df_lin_reg(mice, protocols, females, shuffle=False, seed=np.random.randint(1,1000), save=False, savedir=None, display_fig=False):
    
    delay_list = ['Fixed Delay','Random Delay']
    list_ratios = []
    for group, topi in mice.items():
        for mouse in topi:
            # print(mouse)
    
            gender = 'Female' if mouse in females else 'Male'
    
            for delay in delay_list:
                if delay == 'Fixed Delay':
                    skip_last = False
                else:
                    skip_last = True
                    
                for protocol in protocols:
                    if group == '14' and protocol == 'P0':
                        continue
                    else:
                        
                        print(group, mouse, protocol, delay)
                        
                        # path to the current lick file
                        basedir = fr'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\Behaviour\Group {group}\{mouse} (CM16-Buz - {gender})\{delay}\{protocol}'
                        lick_path = basedir+'\\' + og.file_list(basedir, no_extension=False, ext='.lick')[0]
                        # ot of the valve
                        ot = bv.extract_ot(lick_path.replace('.lick','.param'),skip_last=skip_last)
                        # extract licks
                        licks = bv.load_lickfile(lick_path,ot=ot)
        
                        if display_fig:                
                            plt.figure()
                        
                        if delay == 'Fixed Delay':
                            reward_time = 2.5+(np.mean(ot)/1000)
                        else:
                            reward_time = 2.4+(np.mean(ot)/1000)
                        
                        cm = ['#1f77b4','#ff7f0e']
                        if protocol != 'P0':
                            ns_slope,ns_intercept,ns_auc = lin_reg(licks[licks[:,0]<31],reward_time=reward_time,label='Control',color=cm[0],shuffle=shuffle, seed=seed, display_fig=display_fig)
                            s_slope,s_intercept,s_auc = lin_reg(licks[licks[:,0]>30],reward_time=reward_time,label='Stim',color=cm[1],shuffle=shuffle, seed=seed, display_fig=display_fig)
                        
                            ns_temp = [ns_slope,ns_intercept,ns_auc,mouse,gender,delay, 'No Stim', protocol] # build the dataframe
                            list_ratios.append(ns_temp)
                            s_temp = [s_slope,s_intercept,s_auc,mouse,gender,delay, 'Stim', protocol] # build the dataframe
                            list_ratios.append(s_temp)
                        
                        else:
                            ns_slope,ns_intercept,ns_auc = lin_reg(licks, reward_time=reward_time,label='Control',color=cm[0],shuffle=shuffle, seed=seed, display_fig=display_fig)
                            
                            ns_temp = [ns_slope,ns_intercept,ns_auc,mouse,gender,delay, 'No Stim', protocol] # build the dataframe
                            list_ratios.append(ns_temp)
                            
                        if display_fig:
                            plt.title('{} {} {} [Control: {:.4f}, Stim: {:.4f}] SHUFFLED'.format(mouse,protocol,delay,ns_slope,s_slope))
                            plt.xlabel('Time (s)')
                            plt.ylabel('Density of Licks')
                            plt.legend()
                        
                        if save:
                            plt.savefig(savedir+f'\\individual PSTH with slopes\Fixed and Random exp\{mouse}_{delay}_{protocol}.pdf')


    return pd.DataFrame(list_ratios,columns = ('Slope', 'Intercept', 'AUC', 'Mouse', 'Sex', 'Delay', 'Condition', 'Protocol'))

def plot_average_slope(list_ratios, mice, save=False, save_path=None, density=True, display_fig=True):
    
    slope_nostim_Fixed = list_ratios.loc[(list_ratios['Condition']=='No Stim')&(list_ratios['Delay']=='Fixed Delay')][['Slope', 'Mouse']]
    slope_stim_Fixed = list_ratios.loc[(list_ratios['Condition']=='Stim')&(list_ratios['Delay']=='Fixed Delay')][['Slope', 'Mouse']]
    
    slope_nostim_Random = list_ratios.loc[(list_ratios['Condition']=='No Stim')&(list_ratios['Delay']=='Random Delay')][['Slope', 'Mouse']]
    slope_stim_Random = list_ratios.loc[(list_ratios['Condition']=='Stim')&(list_ratios['Delay']=='Random Delay')][['Slope', 'Mouse']]
    
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
                
                statistic, pvalue = stats.ks_2samp(slope_dic[delay]['nostim']['mean'], slope_dic[delay]['stim']['mean'], alternative='two-sided')
                effectsize = round(abs(mu_nostim-mu_stim)/np.mean([sigma_nostim,sigma_stim]), 3)
                ax.set_title(f'Distribution of the mean slope of the licking (pre reward) per animal.\n{delay}\np_value={pvalue}(two-sample Kolmogorov-Smirnov test)\nEffect size: {effectsize}')
                ax.set_xlabel('slope')
                ax.set_ylabel('probability density')
                ax.legend()
                if save:
                    plt.savefig(fr'{save_path}\ProbDens_LickSlope_{delay}_StimVsNoStim.pdf')
            
            else:
                ax.hist(slope_dic[delay]['nostim']['mean'], alpha=0.5, density=False, color=cm[0], label='No Stim')
               
                ax.hist(slope_dic[delay]['stim']['mean'], alpha=0.5, density=False, color=cm[1], label='Stim')
                
                
                ax.set_title(f'Distribution of the mean slope of the licking (pre reward) per animal.\n{delay}')
                ax.set_xlabel('slope')
                ax.set_ylabel('Count nb')
                ax.legend()
                if save:
                    plt.savefig(fr'{save_path}\LickSlope_{delay}_StimVsNoStim.pdf')  
    
    return slope_dic

def plot_average_slope_by_delay(list_ratios, mice, save=False, save_path=None, density=True, display_fig=True):
    
    slope_nostim_Fixed = list_ratios.loc[(list_ratios['Condition']=='No Stim')&(list_ratios['Delay']=='Fixed Delay')][['Slope', 'Mouse']]
    slope_stim_Fixed = list_ratios.loc[(list_ratios['Condition']=='Stim')&(list_ratios['Delay']=='Fixed Delay')][['Slope', 'Mouse']]
    
    slope_nostim_Random = list_ratios.loc[(list_ratios['Condition']=='No Stim')&(list_ratios['Delay']=='Random Delay')][['Slope', 'Mouse']]
    slope_stim_Random = list_ratios.loc[(list_ratios['Condition']=='Stim')&(list_ratios['Delay']=='Random Delay')][['Slope', 'Mouse']]
    
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
    
    for condition in ['nostim', 'stim']:
        if display_fig:
            fig, ax = plt.subplots(figsize=(8,8))
            cm = ['#1f77b4','#ff7f0e']
            
            if density:
                _, bins_fixe, _ = ax.hist(slope_dic['Fixed Delay'][condition]['mean'], alpha=0.5, density=True, color=cm[0], label='Fixed')
                sigma_fixe = np.std(slope_dic['Fixed Delay'][condition]['mean'])
                mu_fixe = np.mean(slope_dic['Fixed Delay'][condition]['mean'])  
                y_fixe = ((1 / (np.sqrt(2 * np.pi) * sigma_fixe)) *np.exp(-0.5 * (1 / sigma_fixe * (bins_fixe - mu_fixe))**2))
                ax.plot(bins_fixe, y_fixe, color=cm[0]) 
                ax.axvline(mu_fixe, color=cm[0], ls='--')
               
                _, bins_variable, _ = ax.hist(slope_dic['Random Delay'][condition]['mean'], alpha=0.5, density=True, color=cm[1], label='Variable')
                sigma_variable = np.std(slope_dic['Random Delay'][condition]['mean'])
                mu_variable = np.mean(slope_dic['Random Delay'][condition]['mean'])
                y_variable = ((1 / (np.sqrt(2 * np.pi) * sigma_variable)) *np.exp(-0.5 * (1 / sigma_variable * (bins_variable - mu_variable))**2))
                ax.plot(bins_variable, y_variable, color=cm[1])
                ax.axvline(mu_variable, color=cm[1], ls='--')
                
                statistic, pvalue = stats.ks_2samp(slope_dic['Fixed Delay'][condition]['mean'], slope_dic['Random Delay'][condition]['mean'], alternative='two-sided')
                effectsize = round(abs(mu_fixe-mu_variable)/np.mean([sigma_fixe,sigma_variable]), 3)
                ax.set_title(f'Distribution of the mean slope of the licking (pre reward) per animal.\n{condition}\np_value={pvalue}(two-sample Kolmogorov-Smirnov test)\nEffect size: {effectsize}')
                ax.set_xlabel('slope')
                ax.set_ylabel('probability density')
                ax.legend()
                if save:
                    plt.savefig(fr'{save_path}\ProbDens_LickSlope_{condition}_FixeVsVariable.pdf')
            
            else:
                ax.hist(slope_dic[delay]['nostim']['mean'], alpha=0.5, density=False, color=cm[0], label='No Stim')
               
                ax.hist(slope_dic[delay]['stim']['mean'], alpha=0.5, density=False, color=cm[1], label='Stim')
                
                
                ax.set_title(f'Distribution of the mean slope of the licking (pre reward) per animal.\n{delay}')
                ax.set_xlabel('slope')
                ax.set_ylabel('Count nb')
                ax.legend()
                if save:
                    plt.savefig(fr'{save_path}\LickSlope_{delay}_StimVsNoStim.pdf')  
    
    return slope_dic

def stats_maker(slope_dic, savedir):
    stat_dic = {}
    stat_to_save = ['Licking slope from 0 to 2.5s (fixed) or 2.4s (random), Stim Vs No Stim, Paired T-test\n\n']
    for delay, delay_dic in slope_dic.items():
        delay_dic['nostim'].sort_values(by=['mouse'], inplace=True)
        delay_dic['stim'].sort_values(by=['mouse'], inplace=True)
        W,p_value = stats.ttest_rel(delay_dic['nostim']['mean'],delay_dic['stim']['mean'],alternative='two-sided')
        stat_dic[delay] = {'Statistic': W, 'p_value': p_value}
        stat_to_save.append(f'{delay}:\n')
        stat_to_save.append(f'     Statistic: {W}\n')
        stat_to_save.append(f'     P_value: {p_value}\n\n')
    
    with open(f'{savedir}\Stats_LickingSlope_StimVSNoStim.txt', 'w') as stats_file:
            stats_file.writelines(stat_to_save)
        
    return stat_dic

def stats_shufflevsnotshuffle(slope_dic, slope_shuffle_dic, savedir, save=False, seed=100):
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
        with open(f'{savedir}\Stats_WheelSlope_StimvsNoStim.txt', 'w') as stats_file:
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
        list_ratios_shuffle = df_lin_reg(mice,protocols,females, shuffle=True, seed=seed, save=False, savedir=None, display_fig=False)
        slope_shuffle_dic = plot_average_slope(list_ratios_shuffle,mice, save=False, density=True, display_fig=False)

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

def plot_boostrap(boot_dic, boot_dic_shuffle, density=True, save=False):
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
                ax.set_title(f'Bootstrap lick slope distribution, shuffle vs not shuffle,\n{delay} {condition}, \np_value={pvalue}(two-sample Kolmogorov-Smirnov test)\nEffect size: {effectsize}')
                ax.legend()
                if save:
                    plt.savefig(savedir+fr'\bootstrap\100Cycle\\{delay}_{condition}_ShuffleVSNotShuffle.pdf')
    
            else:
                n, bins_notshuffle, _ = ax.hist(condition_dic['endCycle'], alpha=0.5, density=True, color=cm[0], label='Not shuffle')
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
                ax.set_title(f'Bootstrap lick slope distribution, shuffle vs not shuffle,\n{delay} {condition}, \np_value={pvalue}(two-sample Kolmogorov-Smirnov test)\nEffect size: {effectsize}')
                ax.legend()
                
                if save:
                    plt.savefig(savedir+fr'\bootstrap\100Cycle\\density_{delay}_{condition}_ShuffleVSNotShuffle.pdf')

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
                plt.savefig(savedir+fr'\bootstrap\100Cycle\\{condition}_FixedVSVariable.pdf')

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
                plt.savefig(savedir+fr'\bootstrap\100Cycle\\density_{condition}_FixedVSRandom.pdf')

def plot_boostrap_stim_vs_nostim(boot_dic, density=True, savedir=None):
    cm = ['#1f77b4','#ff7f0e']
    for delay in ['Random Delay', 'Fixed Delay']:
        fig, ax = plt.subplots(figsize=(8,8))
        
        nostim_data = boot_dic[delay]['nostim']['endCycle']
        stim_data = boot_dic[delay]['stim']['endCycle']
        if not density:
            ax.hist(nostim_data, color=cm[0], label='No Stim', alpha=0.5)
            ax.hist(stim_data, color=cm[1], label='Stim', alpha=0.5)
            
            statistic, pvalue = stats.ks_2samp(nostim_data, stim_data, alternative='two-sided')
            
            sigma_nostim = np.std(nostim_data)
            mu_nostim = np.mean(nostim_data)  
            sigma_stim = np.std(stim_data)
            mu_stim = np.mean(stim_data)
            effectsize = round(abs(mu_nostim-mu_stim)/np.mean([sigma_nostim,sigma_stim]), 3)
            ax.set_title(f'Bootstrap lick slope distribution, Stim vs not No Stim,\n{delay}, \np_value={pvalue}(two-sample Kolmogorov-Smirnov test)\nEffect size: {effectsize}')
            ax.legend()
            if savedir is not None:
                plt.savefig(savedir+fr'\bootstrap\100Cycle\Stim VS No Stim\\{delay}_StimVSNoStim.pdf')

        else:
            n, bins_nostim, _ = ax.hist(nostim_data, alpha=0.5, density=True, color=cm[0], label='No Stim')
            sigma_nostim = np.std(nostim_data)
            mu_nostim = np.mean(nostim_data)  
            y_nostim = ((1 / (np.sqrt(2 * np.pi) * sigma_nostim)) *np.exp(-0.5 * (1 / sigma_nostim * (bins_nostim - mu_nostim))**2))
            ax.plot(bins_nostim, y_nostim, color=cm[0]) 
            ax.axvline(mu_nostim, color=cm[0], ls='--')
           
            _, bins_stim, _ = ax.hist(stim_data, alpha=0.5, density=True, color=cm[1], label='Stim')
            sigma_stim = np.std(stim_data)
            mu_stim = np.mean(stim_data)
            y_stim = ((1 / (np.sqrt(2 * np.pi) * sigma_stim)) *np.exp(-0.5 * (1 / sigma_stim * (bins_stim - mu_stim))**2))
            ax.plot(bins_stim, y_stim, color=cm[1])
            ax.axvline(mu_stim, color=cm[1], ls='--')
            
            statistic, pvalue = stats.ks_2samp(nostim_data, stim_data, alternative='two-sided')
            effectsize = round(abs(mu_nostim-mu_stim)/np.mean([sigma_nostim,sigma_stim]), 3)
            ax.set_title(f'Bootstrap lick slope distribution, Stim vs No Stim,\n{delay}, \np_value={pvalue}(two-sample Kolmogorov-Smirnov test)\nEffect size: {effectsize}')
            ax.legend()
            
            if savedir is not None:
                plt.savefig(savedir+fr'\bootstrap\100Cycle\Stim VS No Stim\\density_{delay}_StimVSNoStim.pdf')

if __name__ == '__main__':

    savedir = r'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\RESULTS\Behaviour\Anticipation_LinReg\Comparison of slope'
    
    mice = {'14': (6401, 6402, 6409),
            '15': (173, 174, 176),
            '16': (6924, 6928, 6934),
            '17': (6456, 6457)}
    
    protocols = ('P0','P13', 'P15' ,'P16', 'P18')
    females = (6409, 173, 174, 176, 6934, 6456, 6457)
    seed=100
    
    
    # list_ratios = df_lin_reg(mice, protocols, females, shuffle=False, save=False, savedir=savedir, display_fig=False)
    # og.pickle_saving(r'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\RESULTS\Behaviour\Anticipation_LinReg\\Anticipation_df', list_ratios)
    # list_ratios = og.pickle_loading(r'\\equipe2-nas2\Pierre.LE-CABEC\Results\Behaviour\lick\\Anticipation_df')
    
    # slope_dic = plot_average_slope_by_delay(list_ratios, mice, save=True, save_path=savedir, density=True, display_fig=True)
    # og.pickle_saving(r'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\RESULTS\Behaviour\Anticipation_LinReg\\slope_dic', slope_dic)
    # slope_dic = og.pickle_loading(r'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\RESULTS\Behaviour\Anticipation_LinReg\\slope_dic')
    
    # slope_df = None
    # for delay in ['Fixed Delay', 'Random Delay']:
    #     for condition in ['nostim', 'stim']:
    #         print(delay, condition)
    #         print(slope_dic[delay][condition]['mean'].mean())
    #         # temp_df = slope_dic[delay][condition][['mouse', 'mean']]
    #         # condition_list = [condition]*len(temp_df)
    #         # delay_list = [delay]*len(temp_df)
    #         # temp_df['condition'] = condition_list
    #         # temp_df['delay'] = delay_list
    #         # if slope_df is None:
    #         #     slope_df = temp_df
    #         # else:
    #         #    slope_df = pd.concat((slope_df, temp_df), axis=0)
    # print(stats.bartlett(slope_dic['Fixed Delay']['nostim']['mean'], slope_dic['Fixed Delay']['stim']['mean'], slope_dic['Random Delay']['nostim']['mean'], slope_dic['Random Delay']['stim']['mean']))
    
    # formula = 'mean ~ condition + delay + condition:delay'
    # model = ols(formula, data=slope_df).fit()
    # aov_table = sm.stats.anova_lm(model, typ=2, robust='hc3')
    # print(aov_table)
            
    # boot_dic = bootstrap_patterns(slope_dic, run=100, N=11)
    # og.pickle_saving(savedir+r'\bootstrap\100Cycle\\boot_dic', boot_dic)
    boot_dic = og.pickle_loading(r'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\RESULTS\Behaviour\Anticipation_LinReg\Comparison of slope\bootstrap\100Cycle\\boot_dic')

    # boot_dic_shuffle = bootstrap_patterns_shuffle(mice,protocols,females, run=100, N=11)
    # og.pickle_saving(savedir+r'\bootstrap\100Cycle\\boot_dic_shuffle', boot_dic_shuffle)
    boot_dic_shuffle = og.pickle_loading(r'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\RESULTS\Behaviour\Anticipation_LinReg\Comparison of slope\bootstrap\100Cycle\\boot_dic_shuffle')
    
    plot_boostrap(boot_dic, boot_dic_shuffle, density=True, save=False)
    
