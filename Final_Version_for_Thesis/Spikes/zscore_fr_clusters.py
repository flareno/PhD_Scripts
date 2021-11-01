# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 14:19:43 2021

@author: F.LARENO-FACCINI
"""

def my_zscore(col):
    mean_baseline = np.nanmean(col[-15:])
    std_baseline = np.nanstd(col[-15:])
    return (col-mean_baseline)/std_baseline


def random_fixed():
    delays = ('fixed', 'random')
    protocols = ('ns', 'all_stim')
    
    # SHUFFLE ALL UNITS
    all_path = r'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/all_fr_200ms'
    all_df = og.pickle_loading(all_path).T
    shuffle_zscored = []
    for unit in all_df:
        shuffle_zscored.append(my_zscore(all_df[unit]))
    df_z = np.array(shuffle_zscored)
    df_z = df_z[~np.isnan(df_z).any(axis=1),:]
    df_mean = np.nanmean(df_z,axis=0)
    df_sem  = stats.sem(df_z, axis=0,nan_policy='omit')
    print('Total Units:', len(all_df.columns))
    
    a=0
    fig, ax = plt.subplots(2,2, sharex=True)
    ax[0,0].set_ylabel('Firing Rate (z-score)')
    ax[1,0].set_ylabel('Firing Rate (z-score)')
    ax = ax.ravel()
    for delay in delays:
        for protocol in protocols:
            print(delay, protocol)
            labelpath = fr'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/Clustering/Datasets for PCA/Definitive_Labels_HCPC/Labels_{delay}_{protocol}_PYRAMIDAL_70'
            label_df = og.pickle_loading(labelpath)
            # print('Number of units in the class:', len(label_df.index))

            filepath = fr'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/Clustering/Datasets for PCA/{delay}_{protocol}_fr.csv'
            df = pd.read_csv(filepath, index_col='Unnamed: 0').T

            n_clust = label_df['HCPC_Cluster'].max()
            
            for x in range(n_clust+1):
                clust_labels = label_df.loc[label_df['HCPC_Cluster']==x].Cluster_Name
                # print(f'Number of units in group {x}:', len(clust_labels))

                if len(clust_labels)>3:
                    new_df = df[clust_labels]
                    zscored = []
                    for unit in new_df:
                        zscored.append(my_zscore(new_df[unit]))
                
                    mean_z = np.nanmean(np.array(zscored),axis=0)
                    smooth = gaussian_filter1d(mean_z,sigma=1)
                    sem_z = stats.sem(np.array(zscored),axis=0)
                    time = np.linspace(0,9,len(mean_z))
                    ax[a].plot(time, smooth, label=f'Group {x} (n={len(zscored)})')
                    ax[a].fill_between(time, smooth+sem_z, smooth-sem_z, alpha=0.2)
                    
                    ks,pval = ztest(df_mean,mean_z,alternative='smaller')
                    if pval<0.05:
                        print('Group',x, 'p-value:',pval)
                
            ax[a].plot(time, df_mean, color='k', label='Shuffle')
            ax[a].fill_between(time, df_mean+df_sem, df_mean-df_sem, alpha=0.2,color='k')
            ax[a].set_ylim(-2,3)

            ax[a].set_title(f'{delay} {protocol}') 
            if a == 2 or a == 0:
                ax[a].legend(loc='lower right')
            else:
                ax[a].legend(loc='upper right')
            ax[a].axvspan(0,0.5, color='grey', alpha=0.4)
            ax[a].axvspan(1.5,2, color='grey', alpha=0.4)
            if delay=='fixed':
                ax[a].axvspan(2.53,2.68, color='red', alpha=0.4)
            else:
                ax[a].axvspan(2.43,2.93, color='red', alpha=0.4)
            if protocol=='all_stim':
                ax[a].axvspan(1.5,2.5, color='skyblue', alpha=0.4)
 
            a += 1
    fig.suptitle('Z-score firing rate of clusters of neurons')
    plt.show


def plot_protocols():
    delays = ('fixed', 'random')
    protocols = ('P13', 'P15', 'P16','P18')
    # SHUFFLE ALL UNITS
    all_path = r'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/all_fr_200ms'
    all_df = og.pickle_loading(all_path).T
    shuffle_zscored = []
    for unit in all_df:
        shuffle_zscored.append(my_zscore(all_df[unit]))
    df_z = np.array(shuffle_zscored)
    df_z = df_z[~np.isnan(df_z).any(axis=1),:]
    df_mean = np.nanmean(df_z,axis=0)
    df_sem  = stats.sem(df_z, axis=0,nan_policy='omit')

    for delay in delays:
        fig, ax = plt.subplots(2,2, sharex=True)
        
        ax[0,0].set_ylabel('Firing Rate (z-score)')
        ax[1,0].set_ylabel('Firing Rate (z-score)')
        
        ax = ax.ravel()
        for a,protocol in enumerate(protocols):
            print(delay, protocol)
            labelpath = fr'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/Clustering/Datasets for PCA/Definitive_Labels_HCPC/Labels_{delay}_{protocol}_PYRAMIDAL_70'
            label_df = og.pickle_loading(labelpath)
            # print('Number of units in the class:', len(label_df.index))
            filepath = fr'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/Clustering/Datasets for PCA/{delay}_{protocol}_fr.csv'
            df = pd.read_csv(filepath, index_col='Unnamed: 0').T
            
            n_clust = label_df['HCPC_Cluster'].max()
            
            for x in range(n_clust+1):
                clust_labels = label_df.loc[label_df['HCPC_Cluster']==x].Cluster_Name
                # print(f'Number of units in group {x}:', len(clust_labels))

                # if len(clust_labels)>1:
                new_df = df[clust_labels]
                zscored = []
                for unit in new_df:
                    zscored.append(my_zscore(new_df[unit]))
                
                mean_z = np.nanmean(np.array(zscored),axis=0)
                smooth = gaussian_filter1d(mean_z,sigma=1)
                sem_z = stats.sem(np.array(zscored),axis=0)
                time = np.linspace(0,9,len(mean_z))
                ax[a].plot(time, smooth, label=f'Group {x} (n={len(zscored)})')
                ax[a].fill_between(time, smooth+sem_z, smooth-sem_z, alpha=0.2)
            
                ks,pval = ztest(df_mean,mean_z,alternative='larger')
                if pval < 0.05:
                    print('Group',x, 'p-value:',pval)

            ax[a].set_title(f'{delay} {protocol}')
            ax[a].plot(time, df_mean, color='k', label='Shuffle')
            ax[a].fill_between(time, df_mean+df_sem, df_mean-df_sem, alpha=0.2,color='k')
            ax[a].legend()
            ax[a].axvspan(0,0.5, color='grey', alpha=0.4)
            ax[a].axvspan(1.5,2, color='grey', alpha=0.4)
            if delay=='fixed':
                ax[a].axvspan(2.53,2.68, color='red', alpha=0.4)
            else:
                ax[a].axvspan(2.43,2.93, color='red', alpha=0.4)
            ax[a].axvspan(1.5,2.5, color='skyblue', alpha=0.4)
        
        fig.suptitle(f'Z-score firing rate of {delay}')

def slope(col, t_start=0, t_stop =2.5):
    bins = np.arange(0,len(col),1)
    start = int(t_start/(9/len(col)))
    stop = int(t_stop/(9/len(col)))
    slope, *_ = stats.linregress(bins[start:stop], col[start:stop])
    
    return slope

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))

def clusterize_general():
    delays = ('fixed', 'random')
    protocols = ('ns', 'all_stim')
    
    # SHUFFLE ALL UNITS
    all_path = r'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/all_fr_200ms'
    all_df = og.pickle_loading(all_path).T
    # all_df = all_df.sample(frac=1).reset_index(drop=True)
    df_z = np.array(all_df)
    df_z = df_z[~np.isnan(df_z).any(axis=1),:]
    df_mean = np.nanmean(df_z,axis=1)
    print('Total Units:', len(all_df.columns))
    
    t_stop = 2.6
    
    general_df = None
    for delay in delays:
        for protocol in protocols:
            print(delay, protocol)
            labelpath = fr'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/Clustering/Datasets for PCA/Definitive_Labels_HCPC/Labels_{delay}_{protocol}_PYRAMIDAL_70'
            label_df = og.pickle_loading(labelpath)
            # print('Number of units in the class:', len(label_df.index))
    
            filepath = fr'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/Clustering/Datasets for PCA/{delay}_{protocol}_fr.csv'
            df = pd.read_csv(filepath, index_col='Unnamed: 0').T
    
            n_clust = label_df['HCPC_Cluster'].max()
            
            for x in range(n_clust+1):
                clust_labels = label_df.loc[label_df['HCPC_Cluster']==x].Cluster_Name
    
                if len(clust_labels)>2:
                    new_df = df[clust_labels]
                    
                    mean_z = np.nanmean(np.array(new_df),axis=1)
                    
                    # SLOPE
                    slope_mean = slope(mean_z,t_start=0,t_stop=t_stop)
                    # Mean FR
                    start = int(2/(9/len(mean_z)))
                    stop = int(t_stop/(9/len(mean_z)))
                    mean_fr = np.mean(mean_z[start:stop])#/np.mean(mean_z[0:int(1.5/(9/len(mean_z)))])
                    temp__ = pd.DataFrame([slope_mean,mean_fr,f'{delay}_{protocol}_{x}'],index=['Slope','Mean_FR', 'Cluster']).T
                    if general_df is None:
                        general_df = temp__
                    else:
                        general_df = pd.concat([general_df,temp__],axis=0)
    
    # SLOPE
    slope_shuff = slope(df_mean,t_start=0,t_stop=t_stop)
    # Mean FR
    start = int(2/(9/len(df_mean)))
    stop = int(t_stop/(9/len(df_mean)))
    shuff_fr = np.mean(mean_z[start:stop])#/np.mean(mean_z[0:int(1.5/(9/len(mean_z)))])
    shuff__ = pd.DataFrame([slope_shuff,shuff_fr,'Shuffle'],index=['Slope','Mean_FR', 'Cluster']).T
    general_df = pd.concat([general_df,shuff__],axis=0)
    
    general_df['Slope'] = pd.to_numeric(general_df['Slope'])
    general_df['Mean_FR'] = pd.to_numeric(general_df['Mean_FR'])
    
    general_df.plot.scatter('Slope','Mean_FR')
    label_point(general_df.Slope, general_df.Mean_FR, general_df.Cluster, plt)
    
def clusterize_protocols():
    delays = ('fixed', 'random')
    protocols = ('ns','P13', 'P15', 'P16','P18')
    # SHUFFLE ALL UNITS
    all_path = r'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/all_fr_200ms'
    all_df = og.pickle_loading(all_path).T
    # all_df = all_df.sample(frac=1).reset_index(drop=True)
    df_z = np.array(all_df)
    df_z = df_z[~np.isnan(df_z).any(axis=1),:]
    df_mean = np.nanmean(df_z,axis=1)
    
    t_stop = 2.6

    for delay in delays:
        general_df = None
        for a,protocol in enumerate(protocols):
            print(delay, protocol)
            labelpath = fr'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/Clustering/Datasets for PCA/Definitive_Labels_HCPC/Labels_{delay}_{protocol}_PYRAMIDAL_70'
            label_df = og.pickle_loading(labelpath)
            filepath = fr'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/Clustering/Datasets for PCA/{delay}_{protocol}_fr.csv'
            df = pd.read_csv(filepath, index_col='Unnamed: 0').T
            
            n_clust = label_df['HCPC_Cluster'].max()
            
            for x in range(n_clust+1):
                clust_labels = label_df.loc[label_df['HCPC_Cluster']==x].Cluster_Name
    
                if len(clust_labels)>2:
                    new_df = df[clust_labels]
                    
                    mean_z = np.nanmean(np.array(new_df),axis=1)
                    # SLOPE
                    slope_mean = slope(mean_z,t_start=0,t_stop=t_stop)
                    # Mean FR
                    start = int(2/(9/len(mean_z)))
                    stop = int(t_stop/(9/len(mean_z)))
                    mean_fr = np.mean(mean_z[start:stop])#/np.mean(mean_z[0:int(1.5/(9/len(mean_z)))])
                    temp__ = pd.DataFrame([slope_mean,mean_fr,f'{delay}_{protocol}_{x}'],index=['Slope','Mean_FR', 'Cluster']).T
                    if general_df is None:
                        general_df = temp__
                    else:
                        general_df = pd.concat([general_df,temp__],axis=0)
        
        # SLOPE
        slope_shuff = slope(df_mean,t_start=0,t_stop=t_stop)
        # Mean FR
        start = int(2/(9/len(df_mean)))
        stop = int(t_stop/(9/len(df_mean)))
        shuff_fr = np.mean(mean_z[start:stop])#/np.mean(mean_z[0:int(1.5/(9/len(mean_z)))])
        shuff__ = pd.DataFrame([slope_shuff,shuff_fr,'Shuffle'],index=['Slope','Mean_FR', 'Cluster']).T
        general_df = pd.concat([general_df,shuff__],axis=0)
        
        general_df['Slope'] = pd.to_numeric(general_df['Slope'])
        general_df['Mean_FR'] = pd.to_numeric(general_df['Mean_FR'])
        
        general_df.plot.scatter('Slope','Mean_FR')
        label_point(general_df.Slope, general_df.Mean_FR, general_df.Cluster, plt)

###############################################################################
############################################################################### 
import extrapy.Organize as og
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.weightstats import ztest

random_fixed()
# plot_protocols()
# clusterize_general()
# clusterize_protocols()
