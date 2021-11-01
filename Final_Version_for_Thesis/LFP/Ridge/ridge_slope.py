# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:40:52 2021

@author: F.LARENO-FACCINI
"""
import sys
sys.path.append('D:\Pierre.LE-CABEC\Code Pierre')
import numpy as np
import matplotlib.pyplot as plt
import extrapy.Organize as og
import extrapy.Behaviour as bv
import pandas as pd
import scipy.stats as stats
import extrapy.Scalogram as scalogram

def lin_reg(ridge ,bins, reward_time=2.5, display_fig=False, shuffle=False, seed=np.random.randint(0,10000)):

    half_bin = (bins[1]-bins[0])/2 #the bins in the array are not round number we need to take a little bit of margin (half a bin)
    time_lr = np.where(bins<reward_time+half_bin)[0][-1]
    # time_lr = int(reward_time/(9/len(ridge)))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(ridge[:time_lr])
    slope, intercept, *_ = stats.linregress(bins[:time_lr], ridge[:time_lr])
    if display_fig:
        fit = slope*(bins[:time_lr]) + (intercept)
        plt.figure()
        plt.plot(bins, ridge)
        plt.plot(bins[:time_lr], fit, alpha=0.8)
        plt.axvline(reward_time, color='r')
        
    return slope, intercept

def slope_high(df, shuffle=False, seed=np.random.randint(0,10000), display_fig=False):
    df_slope = None
    for delay in df.keys():
        if delay == 'Random Delay':
            reward_time = 2.4
        else:
            reward_time = 2.5
            
        for condition, topi in df[delay].items():
        
            for mouse, ch in topi.items():
                print(mouse, delay, condition)
                if ch is not None:
                    mouse_avg = pd.Series([*ch.values()]).mean() # average all channels of a mouse
                    # print(delay, session, mouse)
                    
                    tfr_sampling_rate = len(mouse_avg)/9
                    start = int(0.5*tfr_sampling_rate)
                    stop = int(8.5*tfr_sampling_rate)
                    time = np.arange(0.5,8.5, 8/len(mouse_avg[start:stop]))
                    # time = np.arange(0,9, 9/len(mouse_avg))

                    slope,_ = lin_reg(mouse_avg[start:stop], time, display_fig=display_fig, reward_time=reward_time, shuffle=shuffle, seed=seed)
                    
                    if df_slope is None:
                        df_slope = pd.DataFrame([slope, delay, condition, mouse], index=['Slope', 'Delay', 'Condition', 'Mouse']).T
                    else:
                        temp__ = pd.DataFrame([slope, delay, condition, mouse], index=['Slope', 'Delay', 'Condition', 'Mouse']).T
                        df_slope = pd.concat([df_slope, temp__])
                        
    df_slope.Slope = df_slope.Slope.astype(float, errors = 'raise')
    df_slope.Mouse = df_slope.Mouse.astype(str, errors = 'raise')
    return df_slope

def slope_low(df_dic, shuffle=False, seed=np.random.randint(0,10000), display_fig=False):
    df_slope = None           
    delays = ('Fixed Delay', 'Random Delay')
    for delay in delays:
        if delay == 'Random Delay':
            reward_time = 2.4
        else:
            reward_time = 2.5
            
        for condition in df_dic.keys():
            for key,data in df_dic[condition].items():
                if delay in key:
                    # print(session, key)
                    tfr_sampling_rate = len(data[:,0])/9
                    start = int(0.5*tfr_sampling_rate)
                    stop = int(8.5*tfr_sampling_rate)
                    time = np.arange(0.5,8.5, 8/len(data[:,0][start:stop]))
    
                    # loop over individual mouse
                    # to compute the ridge of each mouse (on the average power of the mouse)
                    ns_ridge = scalogram.ridge(data)
                    slope,_ = lin_reg(ns_ridge[start:stop], time, display_fig=display_fig, reward_time=reward_time, shuffle=shuffle, seed=seed)
    
                    if df_slope is None:
                        df_slope = pd.DataFrame([slope, key.split('_')[0], condition,  key.split('_')[-1]], index=['Slope', 'Delay', 'Condition', 'Mouse']).T
                    else:
                        temp__ = pd.DataFrame([slope, key.split('_')[0], condition,  key.split('_')[-1]], index=['Slope', 'Delay', 'Condition', 'Mouse']).T
                        df_slope = pd.concat([df_slope, temp__])
                            
    df_slope.Slope = df_slope.Slope.astype(float, errors = 'raise')
    df_slope.Mouse = df_slope.Mouse.astype(str, errors = 'raise')
    return df_slope

def plot_average_slope(df_slope, mice, band, save=False, save_path=None, density=True, display_fig=True):
    
    slope_nostim_Fixed = df_slope.loc[(df_slope['Condition']=='No Stim')&(df_slope['Delay']=='Fixed Delay')][['Slope', 'Mouse']].reset_index(drop=True)
    slope_stim_Fixed = df_slope.loc[(df_slope['Condition']=='Stim')&(df_slope['Delay']=='Fixed Delay')][['Slope', 'Mouse']].reset_index(drop=True)
    
    slope_nostim_Random = df_slope.loc[(df_slope['Condition']=='No Stim')&(df_slope['Delay']=='Random Delay')][['Slope', 'Mouse']].reset_index(drop=True)
    slope_stim_Random = df_slope.loc[(df_slope['Condition']=='Stim')&(df_slope['Delay']=='Random Delay')][['Slope', 'Mouse']].reset_index(drop=True)
    
    slope_dic = {'Fixed Delay': {'nostim': slope_nostim_Fixed, 'stim': slope_stim_Fixed}, 'Random Delay': {'nostim': slope_nostim_Random, 'stim': slope_stim_Random}}
             
    if display_fig:
        for delay in ['Fixed Delay', 'Random Delay']:
            fig, ax = plt.subplots(figsize=(8,8))
            cm = ['#1f77b4','#ff7f0e']
            
            if density:
                _, bins_nostim, _ = ax.hist(slope_dic[delay]['nostim']['Slope'], alpha=0.5, density=True, color=cm[0], label='No Stim')
                sigma_nostim = np.std(slope_dic[delay]['nostim']['Slope'])
                mu_nostim = np.mean(slope_dic[delay]['nostim']['Slope'])  
                y_nostim = ((1 / (np.sqrt(2 * np.pi) * sigma_nostim)) *np.exp(-0.5 * (1 / sigma_nostim * (bins_nostim - mu_nostim))**2))
                ax.plot(bins_nostim, y_nostim, color=cm[0]) 
                ax.axvline(mu_nostim, color=cm[0], ls='--')
               
                _, bins_stim, _ = ax.hist(slope_dic[delay]['stim']['Slope'], alpha=0.5, density=True, color=cm[1], label='Stim')
                sigma_stim = np.std(slope_dic[delay]['stim']['Slope'])
                mu_stim = np.mean(slope_dic[delay]['stim']['Slope'])
                y_stim = ((1 / (np.sqrt(2 * np.pi) * sigma_stim)) *np.exp(-0.5 * (1 / sigma_stim * (bins_stim - mu_stim))**2))
                ax.plot(bins_stim, y_stim, color=cm[1])
                ax.axvline(mu_stim, color=cm[1], ls='--')
                
                statistic, pvalue = stats.levene(slope_dic[delay]['nostim']['Slope'], slope_dic[delay]['stim']['Slope'])
                
                ax.set_title(f"Distribution of the mean {band} Ridge slope  (pre reward) per animal.\n{delay}\np_value={pvalue}(levene's test)")
                ax.set_xlabel('slope')
                ax.set_ylabel('probability density')
                ax.legend()
                if save:
                    plt.savefig(fr'{save_path}\ProbDens_ridgeSlope_{delay}_StimVsNoStim.pdf')
            
            else:
                ax.hist(slope_dic[delay]['nostim']['Slope'], alpha=0.5, density=False, color=cm[0], label='No Stim')
               
                ax.hist(slope_dic[delay]['stim']['Slope'], alpha=0.5, density=False, color=cm[1], label='Stim')
                
                
                ax.set_title(f'Distribution of the mean {band} Ridge slope  (pre reward) per animal.\n{delay}')
                ax.set_xlabel('slope')
                ax.set_ylabel('Count nb')
                ax.legend()
                if save:
                    plt.savefig(fr'{save_path}\{band}_RidgeSlope_{delay}_StimVsNoStim.pdf')  
    
    return slope_dic


def MAD(a,axis=None):
     '''
     Computes median absolute deviation of an array along given axis
     '''
     #Median along given axis but keep reduced axis so that result can still broadcast along a

     med = np.nanmedian(a, axis=axis, keepdims=True)
     mad = np.nanmedian(np.abs(a-med),axis=axis) #MAD along the given axis

     return mad
 
def bootstrap_patterns(slope_dic, run=1000, N=8, input_method='average', output_method='average'):
    
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
            patterns = df['Slope']
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

def bootstrap_patterns_shuffle(df_dic, mice, protocols, females, delays, band, run=1000, N=8, input_method='average', output_method='average'):
    
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
        print(i)
        seed=np.random.randint(1,100000)
        if band == 'Beta' or band == 'Gamma':
            df_slope_shuffle = slope_high(df_dic, shuffle=True, display_fig=False, seed=seed)
        else:
            df_slope_shuffle = slope_low(df_dic, shuffle=True, display_fig=False, seed=seed)
        df_slope_shuffle.reset_index(drop=True, inplace=True)
        
        slope_shuffle_dic = plot_average_slope(df_slope_shuffle, mice, band, save=False, save_path=None, density=True, display_fig=False)

        for delay, delay_dic in slope_shuffle_dic.items():
            for condition, df in delay_dic.items():
                patterns = df['Slope'] 
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

def plot_boostrap(boot_dic, boot_dic_shuffle, band, density=True, save=False, save_path=None):
    
    cm = ['#1f77b4','#ff7f0e']
    for delay, delay_dic in boot_dic.items():
        for condition, condition_dic in delay_dic.items():
            fig, ax = plt.subplots(figsize=(8,8))
            if not density:
                ax.hist(condition_dic['endCycle'], color=cm[0], label='not Shuffle', alpha=0.5)
                ax.hist(boot_dic_shuffle[delay][condition]['endCycle'], color=cm[1], label='Shuffle', alpha=0.5)
                
                sigma_notshuffle = np.std(condition_dic['endCycle'])
                mu_notshuffle = np.mean(condition_dic['endCycle'])  
                sigma_shuffle = np.std(boot_dic_shuffle[delay][condition]['endCycle'])
                mu_shuffle = np.mean(boot_dic_shuffle[delay][condition]['endCycle'])
                effectsize = round(abs(mu_shuffle-mu_notshuffle)/np.mean([sigma_notshuffle,sigma_shuffle]), 3)
                statistic, pvalue = stats.ks_2samp(condition_dic['endCycle'], boot_dic_shuffle[delay][condition]['endCycle'], alternative='two-sided')
                ax.set_title(f'Bootstrap ridge slope distribution, shuffle vs not shuffle,\n{delay} {condition}, \np_value={pvalue}(two-sample Kolmogorov-Smirnov test)\nEffect size: {effectsize}')
                if save:
                    plt.savefig(save_path+fr'\bootstrap\100Cycle\\{band}_{delay}_{condition}_ShuffleVSNotShuffle.pdf')
                ax.legend()
    
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
                ax.set_title(f'Bootstrap ridge slope distribution, shuffle vs not shuffle,\n{delay} {condition}, \np_value={pvalue}(two-sample Kolmogorov-Smirnov test)\nEffect size: {effectsize}')
                ax.legend()
                if save:
                    plt.savefig(save_path+fr'\bootstrap\100Cycle\\{band}_density_{delay}_{condition}_ShuffleVSNotShuffle.pdf')

def plot_boostrap_stimvsnostim(boot_dic, band, density=True, save_path=None):
    
    cm = ['#1f77b4','#ff7f0e']
    for delay, delay_dic in boot_dic.items():
        fig, ax = plt.subplots(figsize=(8,8))
        if not density:
            ax.hist(delay_dic['nostim']['endCycle'], color=cm[0], label='No Stim', alpha=0.5)
            ax.hist(delay_dic['nostim']['endCycle'], color=cm[1], label='Stim', alpha=0.5)
            
            sigma_nostim = np.std(delay_dic['nostim']['endCycle'])
            mu_nostim = np.mean(delay_dic['nostim']['endCycle'])  
            sigma_stim = np.std(delay_dic['stim']['endCycle'])
            mu_stim = np.mean(delay_dic['stim']['endCycle'])
            effectsize = round(abs(mu_stim-mu_nostim)/np.mean([sigma_nostim,sigma_stim]), 3)
            statistic, pvalue = stats.ks_2samp(delay_dic['nostim']['endCycle'], delay_dic['stim']['endCycle'], alternative='two-sided')
            ax.set_title(f'Bootstrap ridge slope distribution, Stim vs No Stim,\n{delay}, \np_value={pvalue}(two-sample Kolmogorov-Smirnov test)\nEffect size: {effectsize}')
            if save_path is not None:
                plt.savefig(save_path+fr'\bootstrap\100cycle\Stim Vs No Stim\\{band}_{delay}_StimVSNoStim.pdf')
            ax.legend()

        else:
            _, bins_nostim, _ = ax.hist(delay_dic['nostim']['endCycle'], alpha=0.5, density=True, color=cm[0], label='No Stim')
            sigma_nostim = np.std(delay_dic['nostim']['endCycle'])
            mu_nostim = np.mean(delay_dic['nostim']['endCycle'])  
            y_nostim = ((1 / (np.sqrt(2 * np.pi) * sigma_nostim)) *np.exp(-0.5 * (1 / sigma_nostim * (bins_nostim - mu_nostim))**2))
            ax.plot(bins_nostim, y_nostim, color=cm[0]) 
            ax.axvline(mu_nostim, color=cm[0], ls='--')
           
            _, bins_stim, _ = ax.hist(delay_dic['stim']['endCycle'], alpha=0.5, density=True, color=cm[1], label='Stim')
            sigma_stim = np.std(delay_dic['stim']['endCycle'])
            mu_stim = np.mean(delay_dic['stim']['endCycle'])
            y_stim = ((1 / (np.sqrt(2 * np.pi) * sigma_stim)) *np.exp(-0.5 * (1 / sigma_stim * (bins_stim - mu_stim))**2))
            ax.plot(bins_stim, y_stim, color=cm[1])
            ax.axvline(mu_stim, color=cm[1], ls='--')
            
            statistic, pvalue = stats.ks_2samp(delay_dic['nostim']['endCycle'], delay_dic['stim']['endCycle'], alternative='two-sided')
            effectsize = round(abs(mu_nostim-mu_stim)/np.mean([sigma_nostim,sigma_stim]), 3)
            
            shapiro_result_stim = stats.shapiro(delay_dic['nostim']['endCycle'])
            shapiro_result_nostim = stats.shapiro(delay_dic['stim']['endCycle'])
            
            variance_result = stats.levene(delay_dic['nostim']['endCycle'], delay_dic['stim']['endCycle'])
            
            
            ax.set_title(f"Bootstrap ridge slope distribution, stim vs no stim,\n{delay}, \np_value={pvalue}(two-sample Kolmogorov-Smirnov test)\nEffect size: {effectsize}\np_value={variance_result[1]}(Levene's test)")
            ax.legend()
            if save_path is not None:
                plt.savefig(save_path+fr'\bootstrap\100cycle\Stim Vs No Stim\\{band}_density_{delay}_StimvsNoStim.pdf')

def plot_boostrap_fixe_vs_random(boot_dic, band, density=True, savedir=None):
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
            ax.set_title(f'Bootstrap rdige slope distribution, Fixed vs not Variable,\n{band} {condition}, \np_value={pvalue}(two-sample Kolmogorov-Smirnov test)\nEffect size: {effectsize}')
            ax.legend()
            if savedir is not None:
                plt.savefig(savedir+fr'\bootstrap\100Cycle\Separated protocol\\{band}_{condition}_FixedVSVariable.pdf')

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
            
            ax.set_title(f'Bootstrap ridge slope distribution, Fixed vs not Variable,\n{band} {condition}, \np_value={pvalue}(two-sample Kolmogorov-Smirnov test)\nEffect size: {effectsize}')
            ax.legend()
            
            if savedir is not None:
                plt.savefig(savedir+fr'\bootstrap\100cycle\Fixed VS Variable\\{band}_density_{condition}_FixedVSRandom.pdf')

###########################################################################################################################
path = r'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\RESULTS\LFP\Ridge\Dataframe\Theta_Good_Trials__tfrSR_40.0'
save_path = r'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\RESULTS\LFP\Ridge\Figures\Slope'

mice = {'14': (6401, 6402, 6409),
            '15': (173, 176),
            '16': (6924,),
            '17': (6456, 6457)}

protocols = ('P0','P13', 'P15' ,'P16', 'P18')
females = (6409, 173, 176, 6456, 6457)
delays = ('Fixed Delay', 'Random Delay')

band = 'Theta'

df_dic = og.pickle_loading(path)
tfr_sampling_rate = 40

if band == 'Beta' or band == 'Gamma':
    df_slope = slope_high(df_dic, shuffle=False, display_fig=False)
else:
    df_slope = slope_low(df_dic, shuffle=False, display_fig=False)
df_slope.reset_index(drop=True, inplace=True)

# og.pickle_saving(save_path+fr'\{band}_Good_Slope', df_slope)
# df_slope = og.pickle_loading(fr'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\RESULTS\LFP\Ridge\Dataframe\bootstrap\{band}_Good_Slope')

slope_dic = plot_average_slope(df_slope, mice, band, save=False, save_path=save_path, density=False, display_fig=True)
# og.pickle_saving(save_path+fr'\{band}_slope_dic', slope_dic)
# slope_dic = og.pickle_loading(fr'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\RESULTS\LFP\Ridge\Dataframe\bootstrap\{band}_slope_dic')

# boot_dic = bootstrap_patterns(slope_dic, run=100, N=8)
# og.pickle_saving(save_path+fr'\bootstrap\{band}_100Cylce', boot_dic)
# boot_dic = og.pickle_loading(fr'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\RESULTS\LFP\Ridge\Dataframe\bootstrap\100 cycle\{band}_100Cycle')


# boot_dic_shuffle = bootstrap_patterns_shuffle(df_dic, mice, protocols, females, delays, band, run=100, N=8)
# og.pickle_saving(save_path+fr'\bootstrap\{band}_shuffle_100Cycle', boot_dic_shuffle)
# boot_dic_shuffle = og.pickle_loading(fr'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\RESULTS\LFP\Ridge\Dataframe\bootstrap\100 cycle\{band}_shuffle_100Cycle')

# plot_boostrap(boot_dic, boot_dic_shuffle, band, density=True, save=False, save_path=save_path)
# plot_boostrap_stimvsnostim(boot_dic, band, density=True, save_path=save_path)
# plot_boostrap_fixe_vs_random(boot_dic, band, density=True, savedir=None)

# for delay in delays:
#     new_df = df_slope.loc[df_slope.Delay == delay]
#     ax = new_df.boxplot(by=['Protocol', 'Condition'], figsize=(12,6))
#     ax.set_title(f'{delay} {band}')
#     ax.set_ylabel('Ridge Slope')
#     plt.savefig(save_path+fr'\{delay.split(" ")[0]}_{band}_Good.pdf')