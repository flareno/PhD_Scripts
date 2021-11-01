# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:28:21 2020
@author: F.LARENO-FACCINI
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import extrapy.Organize as og

def remove_int(df):
    int_labels = [i for i in df.index if 'Interneuron' in i]
    print(len(int_labels), 'Interneurons removed')
    return df.drop(labels=int_labels, inplace=False)

def my_zscore(col):
    mean_baseline = np.nanmean(col[-15:])
    std_baseline = np.nanstd(col[-15:])
    return (col-mean_baseline)/std_baseline

###############################################################################
###############################################################################

delay = 'random'
protocol = 'P13'

# Get Files
path = fr'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/Clustering/Datasets for PCA/{delay}_{protocol}_fr.csv'
df = pd.read_csv(path, index_col='Unnamed: 0')
df = remove_int(df).T

new_df = df.copy()
for unit in new_df:
    df[unit] = my_zscore(new_df[unit])
del new_df


labelpath = fr'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/Clustering/Datasets for PCA/Definitive_Labels_HCPC/Labels_{delay}_{protocol}_PYRAMIDAL'
label_df = og.pickle_loading(labelpath)
labels = label_df.sort_values('HCPC_Cluster',axis=0)['Cluster_Name'].tolist()
df = df[labels]


# new_columns = df.columns[df.loc['2.6'].argsort()]
# df = df[new_columns]



t_stop = 9

extent = (0, t_stop, 0, len(df.columns))
plt.figure()  
im = plt.imshow(df.T, cmap='rainbow', extent=extent, aspect='auto', origin='lower', interpolation='spline16')
plt.colorbar(im)

intervals = [(0,0.5),(1.5,2),(2.53,2.68)]
for x in intervals:
    reward_on = x[0]
    reward_off = x[1]
    
    plt.axvspan(reward_on,reward_off,color='r',alpha=0.2)

cmin,cmax = im.get_clim()
im.set_clim(vmax=cmax/2,vmin=cmin)

plt.title(f'{delay} {protocol} Firing Rate (Z-score)')
plt.ylabel('Clusters')
plt.xlabel('Time (s)')

plt.savefig(fr'D:\F.LARENO.FACCINI\RESULTS\New Results\Spike Sorting\Spike Times\Sorted Spikes\Clustering\Figures\Quantify_Changes\{delay}_{protocol}_Heatmap.pdf')


