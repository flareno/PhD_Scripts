# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 08:40:06 2021

@author: F.LARENO-FACCINI
"""

def sum_and_average(df,freqs):
    samp_period = max(freqs)/len(freqs)
    temp_df = dict.fromkeys(list(df.keys()))
    for mouse, value in df.items():
        for ind,i in enumerate(value):
           df[mouse][ind] = i/np.sum(i) #NORMALIZE
           if ind == 0:
               temp_df[mouse] = [np.sum(i[:int((4-1)/samp_period)]),
                                 np.sum(i[int((4-1)/samp_period):int((10-1)/samp_period)]),
                                 np.sum(i[int((10-1)/samp_period):int((30-1)/samp_period)]),
                                 np.sum(i[int((30-1)/samp_period):int((80-1)/samp_period)])]
           else:
               t__ = [np.sum(i[:int((4-1)/samp_period)]),
                      np.sum(i[int((4-1)/samp_period):int((10-1)/samp_period)]),
                      np.sum(i[int((10-1)/samp_period):int((30-1)/samp_period)]),
                      np.sum(i[int((30-1)/samp_period):int((80-1)/samp_period)])]
               temp_df[mouse] = np.vstack((temp_df[mouse],t__))
        
        temp_df[mouse] = np.nanmean(temp_df[mouse],axis=0) #Average by session
    #Average all mice
    by_band = pd.DataFrame(np.array(list(temp_df.values())).T,index=['Delta','Theta','Beta','Gamma']) 
    mean = by_band.mean(axis=1)
    sem = by_band.sem(axis=1)
    
    mice_df = pd.DataFrame.from_dict(temp_df, orient='index', columns=['Delta', 'Theta', 'Beta', 'Gamma'])
    
    return mean,sem, mice_df


def make_df(basedir):
    delays = ('Fixed', 'Random')   
    sessions = {'No Stim': ['P0', 'P13', 'P15', 'P16', 'P18'],
                'Stim': ['P13', 'P15', 'P16', 'P18']}
    
    all_mice = {}
    mean_df = None
    sem_df = None
    cols = ['Delta', 'Theta', 'Beta', 'Gamma', 'Condition', 'Delay' ]
    
    for delay in delays:
        for cond,values in sessions.items():
            mice = [6401, 6402, 6409, 173, 176, 6924, 6456, 6457]
            temp_df = {key: None for key in mice}
    
            for index, prot in enumerate(values):
                print(cond, prot)
                df = og.pickle_loading(basedir+f'\\Database\{prot}_{cond}_First Half_{delay}_Good')
                print(list(df.keys())[0], df[list(df.keys())[0]].shape)
                # Extract freq bins
                f = df['freqs']
                del(df['freqs'])
                if index == 0:
                    for mouse in list(temp_df.keys()):
                        if mouse in df.keys():
                            temp_df[mouse] = df[mouse]
                else:
                    for mouse in list(temp_df.keys()):
                        if mouse in df.keys():
                            if temp_df[mouse] is not None:
                                temp_df[mouse] = np.concatenate([temp_df[mouse], df[mouse]],axis=0)
                            else:
                                temp_df[mouse] = df[mouse]
                    
                    
                    
            m, s, all_mice[f'{cond}_{delay}'] = sum_and_average(temp_df, f)
    
            mean_temp = m.tolist()
            mean_temp = mean_temp + [cond, delay]
    
            sem_temp = s.tolist()
            sem_temp = sem_temp + [cond, delay]
    
            if mean_df is None:
                mean_df = pd.DataFrame(mean_temp,index=cols).T
                sem_df = pd.DataFrame(sem_temp,index=cols).T
    
            else:
                m__ = pd.DataFrame(mean_temp,index=cols).T
                s__ = pd.DataFrame(sem_temp,index=cols).T
                mean_df = pd.concat([mean_df,m__],ignore_index=True)
                sem_df = pd.concat([sem_df,s__],ignore_index=True)
    
    bands = ('Delta', 'Theta', 'Beta', 'Gamma')
    
    for band in bands:
        mean_df[band] = mean_df[band].astype(float, errors = 'raise')
        sem_df[band] = sem_df[band].astype(float, errors = 'raise')

    return mean_df, sem_df, all_mice


def remove_uneven_animals(ns, s):
    index_ns = list(ns.index)
    index_s = list(s.index)
    
    list1_as_set = set(index_ns)
    to_keep = list(list1_as_set.intersection(index_s))
    
    return ns.T[to_keep].T, s.T[to_keep].T


##############################################################################
##############################################################################
import numpy as np
import extrapy.Organize as og
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


basedir = r'D:\F.LARENO.FACCINI\RESULTS\New Results\LFP\Welch_Half_Trial'

mean_df, sem_df, all_values = make_df(basedir)





delays = ('Fixed', 'Random')   
bands = ('Delta', 'Theta', 'Beta', 'Gamma')


# # CONTROL vs STIM
# for key in all_values.keys():
#     if 'No Stim' in key:
#         # print(key)
#         ns,s = remove_uneven_animals(all_values[key], all_values[key.replace('No Stim', 'Stim')])
        
#         for band in bands:
#             st, pval = stats.ttest_rel(ns[band], s[band])
#             # if pval < 0.05:
#             print(key, band, 'p-value: ', pval)
        
# FIXED vs RANDOM
for key in all_values.keys():
    if 'Fixed' in key:
        ns,s = remove_uneven_animals(all_values[key], all_values[key.replace('Fixed', 'Random')])
        # print(key,len(ns))
        
        ns.plot.box(ylabel="Normalized Power (%)",
                  colormap='tab20',
                  rot=0,
                  title=f'{key} FIXED')
        # plt.savefig(fr'D:\F.LARENO.FACCINI\RESULTS\New Results\LFP\Welch_Half_Trial\Figures\box\Fixed_{key}.pdf')
        
        s.plot.box(ylabel="Normalized Power (%)",
                  colormap='tab20',
                  rot=0,
                  title=f'{key} RANDOM')
        # plt.savefig(fr'D:\F.LARENO.FACCINI\RESULTS\New Results\LFP\Welch_Half_Trial\Figures\box\Random_{key}.pdf')

        
        for band in bands:
            st, pval = stats.ttest_rel(ns[band], s[band], alternative='greater')
            
            effectsize = og.effectsize(ns[band],s[band])
            print('ES', key,band, effectsize)
            if band == 'Delta':
                print('------------------')
                print(key)
                print('Fixed', ns[band].mean(), ns[band].sem())
                print('Random', s[band].mean(), s[band].sem())

            if pval < 0.05:
                print(key, band, 'p-value: ', pval)
            
            # # SHAPIRO
            # print(key, band)
            # print(stats.shapiro(ns[band]))
            # print(stats.shapiro(s[band]))
