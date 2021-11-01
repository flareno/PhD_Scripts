# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:49:34 2021

@author: F.LARENO-FACCINI
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import extrapy.Organize as og
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from matplotlib import cm
from matplotlib import colors
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler 

pd.set_option("display.precision", 8)

def wf_params():
    delays = ('Fixed', 'Random')
    list_values = []
    full_wf = None
    for delay in delays:
        basedir = fr'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Waveforms/{delay} Delay'
        files = og.file_list(basedir, False)
        
        for file in files:
            name = file.split('_')[0]
            
            path = basedir+'/'+file
            df = pd.read_excel(path,sheet_name=None, index_col=0)
            info = df['info'].loc[df['info']['cell_label']>=0]
            info = info.reset_index()
            del df['info']
        
        
            for i, (cluster, values) in enumerate(df.items()):
                print(name, cluster)
                a = np.array(values)
                max_ind = np.unravel_index(np.argmax(a), a.shape)
                temp__ = a[:,max_ind[1]]
                
                if not np.isnan(temp__).all():
                    x = pd.DataFrame(temp__, columns=[f'{name}_{cluster.split(" ")[-1]}'])
                    if full_wf is None:
                        full_wf = x
                    else:
                        full_wf = pd.concat([full_wf,x],axis=1)
                    
                    number_peaks = info['nb_peak'][i]
                    
                    # Peak-to-peak (in ms)
                    peak_amp = np.min(temp__)                    
                    # down_peak = (np.where(temp__ == np.min(temp__))[0][0]/20000)*1000
                    # up_peak = (np.where(temp__ == np.max(temp__[np.where(temp__ == np.min(temp__))[0][0]:]))[0][0]/20000)*1000
                    # peak_to_peak = (up_peak-down_peak)
                    down_peak = np.min(temp__)                    
                    up_peak = np.max(temp__[np.where(temp__ == np.min(temp__))[0][0]:])
                    peak_to_peak = (up_peak+np.abs(down_peak))


                    # Half-width
                    time = np.linspace(0,120/20000,120)
                    spline = UnivariateSpline(time,temp__-np.min(temp__)/2, s=0)
                    r1, r2 = spline.roots() # find the roots
                    half_width = (r2-r1)*1000
                    
                    # peak_to_peak = np.abs(peak_amp + (-np.max(temp__)))
                    
                    temp_df = [peak_to_peak,half_width,number_peaks,f'{name}_{cluster.split(" ")[-1]}', delay]  
                    list_values.append(temp_df)
    
    return pd.DataFrame(list_values,columns = ['Peak_to_Peak','Half Width', 'Number_Peaks', 'Cluster Name', 'Delay']), full_wf # dataframe columns names


wf_df, all_wf = wf_params()
colori = np.where((wf_df["Peak_to_Peak"]<15)&(wf_df["Half Width"]<0.20),'tab:blue','tab:green')
wf_df.plot.scatter(y = 'Peak_to_Peak', x = 'Half Width', c=colori)
# all_wf.to_excel(r'D:\F.LARENO.FACCINI\RESULTS\New Results\Spike Sorting\Waveforms\all_wf.xlsx',index=None)

list_labels = []
for i,v in enumerate(wf_df.Peak_to_Peak):
    if v<15 and wf_df['Half Width'][i] < 0.2:
        list_labels.append('Interneuron')
    else:
        list_labels.append('Pyramidal')

list_labels = pd.DataFrame(list_labels, columns=['Cell Type'])
wf_df = pd.concat([wf_df,list_labels], axis=1)

# wf_df.to_excel(r'D:\F.LARENO.FACCINI\RESULTS\New Results\Spike Sorting\Waveforms\variables_for_pca_wf.xlsx',index=None)


pyr_df = wf_df[wf_df['Cell Type'] == 'Pyramidal']
pyr_df.reset_index(inplace=True)
int_df = wf_df[wf_df['Cell Type'] == 'Interneuron']
int_df.reset_index(inplace=True)

# =============================================================================
# HIERARCHICAL CLUSTERING
# =============================================================================

scaler = StandardScaler()
center_norm_matrix = scaler.fit_transform(pyr_df[['Peak_to_Peak', 'Half Width']])

N_CLUST = 4

#HAC-PCA based with n_components and n_clusters settings
HCPC = AgglomerativeClustering(n_clusters=N_CLUST, linkage='ward')
HCPC_clusters = HCPC.fit_predict(center_norm_matrix)

clr = []
cmap = cm.get_cmap('jet', N_CLUST)
for c in range(cmap.N):
    rgba = cmap(c)
    clr.append(colors.rgb2hex(rgba))

plt.figure()
for i in range(len(pyr_df['Peak_to_Peak'])):
    plt.scatter(center_norm_matrix[i][0], center_norm_matrix[i][1], color=clr[HCPC_clusters[i]], alpha=0.5, label=f'Cluster {HCPC_clusters[i]}')

plt.title('HCPC')
plt.axvline(x=0.0,color='k',linestyle='--')
plt.axhline(y=0.0,color='k',linestyle='--')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())


hcpc_df = pd.DataFrame(HCPC_clusters.T, columns=['HCPC Cluster'], index=pyr_df['Cluster Name'])
clustered_df = pd.concat([pyr_df,hcpc_df], axis=1)
    

fig, ax = plt.subplots(1,N_CLUST+2,sharey=True)
for e in range(N_CLUST):
    time = np.linspace(0,120/20000,120)
    prova__ = clustered_df.loc[clustered_df['HCPC Cluster'] == e]    
    temp_index = prova__.index
    
    new_df = all_wf[temp_index]
    for i in new_df:
        print(i)
        ax[e].plot(time, new_df[i], color='k', alpha=0.2)
    
    ax[N_CLUST+1].plot(time,np.mean(new_df,axis=1),color=clr[e])
    ax[e].plot(time,np.mean(new_df,axis=1),color='r')
    ax[e].set_title(f'Cluster {e}')

all_int = all_wf[int_df['Cluster Name']]
for d in all_int:
    ax[N_CLUST].plot(time,all_int, color='k', alpha=0.2)

ax[N_CLUST].plot(time,np.mean(all_int,axis=1),color='r')
ax[N_CLUST+1].plot(time,np.mean(all_int,axis=1),color='green')
ax[N_CLUST].set_title('Interneurons')

# new_list = []
# for ind,x in enumerate(HCPC_clusters):
#     t = 'Pyramidal' if x == 0 else 'Interneuron'
#     new_list.append(t)

# new_list = pd.DataFrame(new_list, columns=['Cell Type'])
# wf_df = pd.concat([wf_df,new_list], axis=1)
# # wf_df.to_excel(r'D:\F.LARENO.FACCINI\RESULTS\New Results\Spike Sorting\Waveforms\parameters_wf.xlsx',index=None)

