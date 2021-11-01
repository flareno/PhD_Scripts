# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 13:23:30 2021

@author: F.LARENO-FACCINI
"""

import extrapy.Behaviour as bv
import scipy.stats as stats
import matplotlib.pyplot as plt
import extrapy.Organize as og
import numpy as np
import sklearn.metrics as metrics
import pandas as pd

def lin_reg(licks, label, color, reward_time=2.5, samp_period = 0.05,shuffle=False):
    if len(licks) >0:
        time_lr = int(reward_time/samp_period)
        # print('time',time_lr)
        n,bins,*_ = bv.psth_lick(licks,samp_period=samp_period,density=True, label=label, color=color)
        if shuffle:
            np.random.shuffle(n)
        slope, intercept, *_ = stats.linregress(bins[:time_lr], n[:time_lr])
        fit = slope*(bins[:time_lr]) + (intercept)
        plt.plot(bins[:time_lr], fit, alpha=0.8, color=color)
        
        auc = metrics.auc(bins[:time_lr], n[:time_lr])
        # print(len(bins[:time_lr]), len(n[:time_lr]))
        # print(slope,intercept,auc)
        
        return slope, intercept, auc
    else:
        return np.nan, np.nan, np.nan

def df_lin_reg(mice, protocols, females,savedir,shuffle=False):
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
                    print(group, mouse, protocol, delay)
                    
                    # path to the current lick file
                    basedir = fr'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\Behaviour\Group {group}\{mouse} (CM16-Buz - {gender})\{delay}\{protocol}'
                    lick_path = basedir+'\\' + og.file_list(basedir, no_extension=False, ext='.lick')[0]
                    # ot of the valve
                    ot = bv.extract_ot(lick_path.replace('.lick','.param'),skip_last=skip_last)
                    # extract licks
                    licks = bv.load_lickfile(lick_path,ot=ot)
    
                    plt.figure()
                    if delay == 'Fixed Delay':
                        reward_time = 2.5+(np.mean(ot)/1000)
                    else:
                        reward_time = 2.4+(np.mean(ot)/1000)
                    
                    cm = ['#1f77b4','#ff7f0e']
                    
                    ns_slope,ns_intercept,ns_auc = lin_reg(licks[licks[:,0]<31],reward_time=reward_time,label='Control',color=cm[0],shuffle=shuffle)
                    s_slope,s_intercept,s_auc = lin_reg(licks[licks[:,0]>30],reward_time=reward_time,label='Stim',color=cm[1],shuffle=shuffle)
                    
                    plt.title('{} {} {} [Control: {:.4f}, Stim: {:.4f}] SHUFFLED'.format(mouse,protocol,delay,ns_slope,s_slope))
                    plt.xlabel('Time (s)')
                    plt.ylabel('Density of Licks')
                    plt.legend()
                
                    ns_temp = [ns_slope,ns_intercept,ns_auc,mouse,gender,delay, 'No Stim', protocol] # build the dataframe
                    list_ratios.append(ns_temp)
                    s_temp = [s_slope,s_intercept,s_auc,mouse,gender,delay, 'Stim', protocol] # build the dataframe
                    list_ratios.append(s_temp)
                    # plt.savefig(savedir+f'\\individual PSTH with slopes\Fixed and Random exp\{mouse}_{delay}_{protocol}_Shuffled.pdf')
                    plt.close()


    return pd.DataFrame(list_ratios,columns = ('Slope', 'Intercept', 'AUC', 'Mouse', 'Sex', 'Delay', 'Condition', 'Protocol'))


if __name__ == '__main__':

    savedir = r'D:\F.LARENO.FACCINI\RESULTS\Behaviour\Anticipation_LinReg\Shuffled'
    
    mice = {'14': (6401, 6402, 6409),
            '15': (173, 174, 176),
            '16': (6924, 6928, 6934),
            '17': (6456, 6457)}
    
    protocols = ('P13', 'P15' ,'P16', 'P18')
    females = (6409, 173, 174, 176, 6934, 6456, 6457)
    
    
    list_ratios = df_lin_reg(mice, protocols, females, savedir,shuffle=False)
    
    # og.pickle_saving(savedir+'\\Anticipation_df_Shuffled', list_ratios)
    
    slope_df = None
    for _,topi in mice.items():
        for mouse in topi:
            for delay in ('Fixed Delay', 'Random Delay'):
                for cond in ('No Stim', 'Stim'):
                    temp_slope = list_ratios[(list_ratios.Delay == delay) & (list_ratios.Condition == cond) & (list_ratios.Mouse == mouse)].Slope.mean()
                    temp_df = pd.DataFrame([{'Slope': temp_slope,
                                             'Mouse': mouse,
                                             'Delay': delay,
                                             'Condition': cond}],index=None)
                    if slope_df is None:
                        slope_df = temp_df
                    else:
                        slope_df = pd.concat((slope_df,temp_df),axis=0)
                    
    for delay in ('Fixed Delay', 'Random Delay'):
        
        ns = slope_df[(slope_df.Delay == delay) & (slope_df.Condition == 'No Stim')]#.Slope
        s = slope_df[(slope_df.Delay == delay) & (slope_df.Condition == 'Stim')]#.Slope

        df_to_plot = pd.concat((ns,s),axis=0)
        df_to_plot.boxplot(column='Slope', by='Condition')
        plt.title(delay)
        print(ns.Slope.mean(),s.Slope.mean())
        print(delay)
        print(stats.ttest_rel(ns.Slope,s.Slope))
        print(og.effectsize(ns.Slope, s.Slope))



    for cond in ('No Stim', 'Stim'):
        ns = slope_df[(slope_df.Delay == 'Fixed Delay') & (slope_df.Condition == cond)]#.Slope
        s = slope_df[(slope_df.Delay == 'Random Delay') & (slope_df.Condition == cond)]#.Slope
        

        df_to_plot = pd.concat((ns,s),axis=0)
        df_to_plot.boxplot(column='Slope', by='Delay')
        plt.title(cond)
        print(ns.Slope.mean(),s.Slope.mean())
        print(cond)
        print(stats.ttest_rel(ns.Slope,s.Slope))
        print(og.effectsize(ns.Slope, s.Slope))

