# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:20:05 2021

@author: F.LARENO-FACCINI
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import extrapy.Organize as og
import Clustering
from sklearn.metrics import auc as auc

def slope_auc(col, t_stop =2.5):
    bins = np.arange(0,len(col),1)
    time_lr = int(t_stop/(9/len(col)))
    
    slope, *_ = stats.linregress(bins[:time_lr], col[:time_lr])
    
    return slope

def zscore_peak(col, tstart=1.5, tstop=4):

    mean_baseline = np.mean(col[-20:])
    std_baseline = np.std(col[-20:])
    zscored = (col-mean_baseline)/std_baseline

    start = int(tstart/(9/len(col)))
    stop = int(tstop/(9/len(col)))
    pox_peaks = (np.max(zscored[start:stop]), np.min(zscored[start:stop]))
    
    return pox_peaks[np.argmax(np.abs(pox_peaks))]

def find_auc(col):
    path = r'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/all_spiketrains'
    file = og.pickle_loading(path)

    name = col.name
    print(name)
    delay,state,cell,cluster,unit = name.split('_')
    
    spiketrains = file[f'{delay}_{state}_{cell}'][f'{cluster}_{unit}']
    
    temp__ = None
    for train in spiketrains:
        if temp__ is None:
            temp__ = np.array(train)
        else:
            temp__ = np.hstack((temp__,np.array(train)))
    temp__ = np.sort(temp__)
    plt.figure()
    n,bins,_ = plt.hist(temp__,bins=180,density=True)
    plt.close()
    times = ((0,0.5),(1.5,2),(2.2,3.5))
    aucs = []
    for i in times:
        start = int(i[0]/(9/len(n))) 
        stop = int(i[1]/(9/len(n))) 
        aucs.append(auc(bins[start:stop],n[start:stop]))
    return(aucs)

#################################################################################################
#################################################################################################
#################################################################################################

protocol = 'P13'
delay = 'fixed'

path = fr'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/Clustering/Datasets for PCA/{delay}_{protocol}_fr.csv'
df = pd.read_csv(path,index_col='Unnamed: 0').T


all_values = None
for x in df:
    if len(df[df[x]>0]) > 1:
        # SLOPE
        slope = slope_auc(df[x])
        # MEAN FIRING RATE
        mean = df[x].mean()
        # TIME OF PEAK
        peak_time = float(df[x].idxmax())
        # ZSCORED PEAK AT REWARD
        max_peak = zscore_peak(df[x])
        # AUC
        auc_cue1,auc_cue2,auc_reward = find_auc(df[x])
    
        if all_values is None:
            all_values = pd.DataFrame([slope,mean,peak_time,max_peak,auc_cue1,auc_cue2,auc_reward], index=['Slope','Mean_FR','Time_Peak','Peak_Reward','AUC_Cue1','AUC_Cue2','AUC_Reward'], columns=[x]).T
        else:
            temp__ = pd.DataFrame([slope,mean,peak_time,max_peak,auc_cue1,auc_cue2,auc_reward], index=['Slope','Mean_FR','Time_Peak','Peak_Reward','AUC_Cue1','AUC_Cue2','AUC_Reward'], columns=[x]).T
            all_values = pd.concat([all_values,temp__],axis=0)
        
all_values.to_csv(fr'D:\F.LARENO.FACCINI\RESULTS\New Results\Spike Sorting\Spike Times\Sorted Spikes\Clustering\Datasets for PCA\new pca\{delay}_{protocol}_values_pca.csv')

# N_CLUST = 4

# principalDf, principalComponents, ExpVar = Clustering.do_pca(all_values,3)
# hcpc_df,HCPC_clusters = Clustering.do_hcpc(all_values, principalDf, principalComponents, ExpVar, N_CLUST, protocol, delay)
# Clustering.plot_hcpc(all_values, hcpc_df,HCPC_clusters, ExpVar,delay,protocol,N_CLUST)
    

   


# fig, ax = plt.subplots(2,2)
# fig.suptitle(f'{delay} {protocol}')

# for x in range(N_CLUST):
#     clust_labels = hcpc_df.loc[hcpc_df['HCPC_Cluster']==x].Cluster_Name
#     new_df = df[clust_labels].T

#     # print(f'CLUSTER: {x}', '\n', clust_labels)
#     mean = np.mean(new_df,axis=0)
#     sem = stats.sem(new_df,axis=0)
#     time = np.linspace(0,9,len(mean))

#     ax = ax.ravel()

#     ax[x].plot(time, mean)
#     ax[x].fill_between(time, mean+sem, mean-sem, alpha=0.2)
#     ax[x].axvspan(0, 0.5,color='g', alpha = 0.2) # Cue1
#     ax[x].axvspan(1.5, 2,color='g', alpha = 0.2) #Cue2
#     ax[x].axvline(1.03, color='r', linestyle='--')
#     ax[x].axvline(2.53, color='r', linestyle='--')
#     ax[x].axvspan(2.43,2.93, color='r', alpha=0.2)
#     ax[x].set_title(f'Cluster {x}')



# # all_values.plot.scatter(x='Peak_Reward', y='AUC_Reward')
