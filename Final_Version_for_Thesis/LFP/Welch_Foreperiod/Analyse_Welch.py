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

    segments = ('First Half','Second Half')
    delays = ('Fixed', 'Random')   
    protocols = {'NB': ['No Stim'],
                'P0': ['No Stim'],
                'P13': ['No Stim', 'Stim'],
                'P15': ['No Stim', 'Stim'],
                'P16': ['No Stim', 'Stim'],
                'P18': ['No Stim', 'Stim']}
    
    all_mice = {}
    mean_df = None
    sem_df = None
    cols = ['Delta', 'Theta', 'Beta', 'Gamma', 'Protocol', 'Condition', 'Segment', 'Delay' ]
    
    for delay in delays:
        for protocol,values in protocols.items():
            for condition in values:
                for segment in segments:
                    df = og.pickle_loading(basedir+f'\\Database\{protocol}_{condition}_{segment}_{delay}_Good')
                    # Extract freq bins
                    f = df['freqs']
                    del(df['freqs'])
                    
                    m, s, all_mice[f'{protocol}_{condition}_{segment}_{delay}'] = sum_and_average(df, f)
        
                    mean_temp = m.tolist()
                    mean_temp = mean_temp + [protocol, condition, segment, delay]
        
                    sem_temp = s.tolist()
                    sem_temp = sem_temp + [protocol, condition, segment, delay]
        
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


# ax = mean_df.plot(kind='bar',
#           ylabel="Normalized Power (%)",
#           colormap='tab20',
#           rot=0,
#           # yerr=sem_df,
#           # width=0.8,
#           title='Comparison of normalized power by bands across different sessions')
# ax.set_xticklabels(mean_df.Protocol)



prots = ('NB', 'P0', 'P13', 'P15', 'P16', 'P18')
delays = ('Fixed', 'Random')   
bands = ('Delta', 'Theta', 'Beta', 'Gamma')

# for protocol in prots:
#     for delay in delays:
#         new_df_mean = mean_df.loc[(mean_df.Protocol == protocol) & (mean_df.Delay == delay)]
#         new_df_sem = sem_df.loc[(sem_df.Protocol == protocol) & (sem_df.Delay == delay)]
        
#         ax = new_df_mean.plot(kind='bar',
#                   ylabel="Normalized Power (%)",
#                   colormap='tab20',
#                   rot=0,
#                   yerr=new_df_sem,
#                   width=0.8,
#                   title=f'Normalized power by bands {protocol} {delay}')
#         ax.set_xticklabels(new_df_mean.Condition + ' ' + new_df_mean.Segment)
                

# # CONTROL vs STIM
# for key in all_values.keys():
#     if '_Stim' in key and 'First' in key:
#         # print(key)
#         ns,s = remove_uneven_animals(all_values[key.replace('Stim', 'No Stim')], all_values[key])
        
#         for band in bands:
#             st, pval = stats.ttest_rel(ns[band], s[band])
#             if pval < 0.05:
#                 print(key, band, 'p-value: ', pval)
        
# FIXED vs RANDOM
for key in all_values.keys():
    if 'Fixed' in key and 'First' in key:
        ns,s = remove_uneven_animals(all_values[key], all_values[key.replace('Fixed', 'Random')])
        # print(key,len(ns))
        
        ns.plot.box(ylabel="Normalized Power (%)",
                  colormap='tab20',
                  rot=0,
                  title=key)
        # plt.savefig(fr'D:\F.LARENO.FACCINI\RESULTS\New Results\LFP\Welch_Half_Trial\Figures\box\Fixed_{key}.pdf')
        
        s.plot.box(ylabel="Normalized Power (%)",
                  colormap='tab20',
                  rot=0,
                  title=key)
        # plt.savefig(fr'D:\F.LARENO.FACCINI\RESULTS\New Results\LFP\Welch_Half_Trial\Figures\box\Random_{key}.pdf')

        
        for band in bands:
            st, pval = stats.ttest_rel(ns[band], s[band], alternative='greater')
            
            # effectsize = abs((np.mean(ns[band])-np.mean(s[band]))/np.mean([np.std(ns[band]), np.std(s[band])]))
            # print('ES', key,band, effectsize)
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
