# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:03:20 2020

@author: F.LARENO-FACCINI
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import extrapy.Behaviour as bv
import seaborn as sns

mouse = 6409
path = f"D:/F.LARENO.FACCINI/Preliminary Results/Behaviour/Database/Group14_{mouse}.xlsx"
df = pd.read_excel(path, sheet_name=None)
keys = list(df.keys())
savedir = "D:\F.LARENO.FACCINI\Preliminary Results\Behaviour\Figures"

for e in keys:
    if 'Envelope' in e: 
        ns = df[e]
            
        for (columnName,columnData) in ns.iteritems():
            ns[columnName] = (columnData/np.max(columnData)) # Normalize licks by trial

        # Bins
        bins = np.asarray(df['Bins of TOT'])       
        # Normalized n
        norm_n = ns.set_index(bins[:-1,0]) # Set the bins as index so that when plotting it has the right time scale
        
        # Get the opening time of the valve (ot) for each session.
        # If there are more values per session, it takes the most frequent one (the valuse used in most of the trials).
        ot = []
        for i in norm_n.columns:
            try:
               ot.append(np.argmax(np.bincount(df[f'Session {i}']['Opening Time (ms)'])))
            except:
                ot.append(np.nan)
        ot = np.asarray(ot)
        reward = 2+ot/1000+0.9 # Precise time of the release of the reward for each session
        
        # Convolve the lick times
        envelope = np.empty((norm_n.shape[0],norm_n.shape[1]))
        fig1,ax = plt.subplots(1,1)
        for idx, (columnName,columnData) in enumerate(norm_n.iteritems()):
            envelope[:,idx] = bv.envelope(norm_n[columnName],bins,sigma=2,label=columnName,y_label='normalized frequency of lick',ax=ax)
            # plt.axvline(reward[idx])
        fig1.legend(fontsize="x-small")
        ax.set_title(f'{e}')
        fig1.tight_layout()
        fig1.savefig(savedir+f"\LinePlot\{mouse}_{e}.pdf")
        
        fig2,axes = plt.subplots(1,1)
        axes.set_title(f'{e}')
        sns.heatmap(norm_n.T,xticklabels=100, yticklabels=True, cmap='viridis')#,robust=True)
        axes.set_xlabel('Time (s)')
        fig2.tight_layout()
        fig2.savefig(savedir+f"\Heatmap\{mouse}_{e}.pdf")

        plt.close('all')

