# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 19:40:33 2021

@author: F.LARENO-FACCINI
"""

import extrapy.Organize as og
import matplotlib.pyplot as plt
import seaborn as sns


def plot_experiment(df, delays, what, savedir=None, save=False, violin=False, box=True):
    sexes = ('Male','Female')
    # for sex in sexes:
    for delay in delays:
        if violin:
            fig, ax = plt.subplots(2,2,sharey=True,sharex=True)
            ax=ax.ravel()
            for ind, protocol in enumerate(protocols):
                new_df = df.loc[(df['Delay']==delay) & (df['Protocol']==protocol)]# & (df['Sex'] == sex)]
                # print(delay, len(new_df))
                sns.violinplot(x=new_df['Condition'], y=new_df[what],ax=ax[ind])
                fig.suptitle(f'{what} for delay of {delay}')
                ax[ind].set_title(protocol)
            if save:
                fig.savefig(savedir+f'\Violin_{what}_{delay}.pdf')
                plt.close()

        if box:
            fig2,ax2= plt.subplots(2,2,sharey=True,sharex=True)
            ax2=ax2.ravel()
            
            for ind, protocol in enumerate(protocols):
                new_df = df.loc[(df['Delay']==delay) & (df['Protocol']==protocol)]# & (df['Sex'] == sex)]
                # print(delay, len(new_df))
                sns.boxplot(x=new_df['Condition'], y=new_df[what],ax=ax2[ind])
                fig2.suptitle(f'{what} for delay of {delay}')
                ax2[ind].set_title(protocol)
            if save:
                fig2.savefig(savedir+f'\BoxPlot_{what}_{delay}.pdf')
                plt.close()

def plot_training(df, what, savedir=None, save=False, violin=False, box=True):
    sexes = ('Male','Female')
    # for sex in sexes:

        # new_df = df.loc[df['Sex'] == sex]
        # print(delay, len(new_df))
        # if len(new_df)>1:
    if violin:
        # Violin Plot
        plt.figure(figsize=(15,7))
        sns.violinplot(x=df['Session'], y=df[what])
        plt.title(f'{what} training')
        if save:
            plt.savefig(savedir+f'\Violin_{what}_Training.png')
            plt.close()
    if box:
        # Box Plot
        plt.figure(figsize=(15,7))
        sns.boxplot(x=df['Session'], y=df[what])
        plt.title(f'{what} training')
        if save:
            plt.savefig(savedir+f'\BoxPlot_{what}_Training.png')
            plt.close()



what = 'AUC'

savedir = fr'D:/F.LARENO.FACCINI/RESULTS/Behaviour/Anticipation_LinReg/Comparison of {what}'#/by sex'
path = r'D:/F.LARENO.FACCINI/RESULTS/Behaviour/Anticipation_LinReg/Anticipation_df'
df = og.pickle_loading(path)

protocols = ('P13', 'P15' ,'P16', 'P18')
delays = ['Fixed Delay','Random Delay']

plot_experiment(df,delays,what,savedir,save=True)   

# plot_training(df,what,savedir,save=True)




   
'''
# =============================================================================
# Slope*AUC
# =============================================================================

for delay in delays:
    fig2,ax2= plt.subplots(2,2,sharey=True,sharex=True)
    ax2=ax2.ravel()
    
    for ind, protocol in enumerate(protocols):
        new_df = df.loc[(df['Delay']==delay) & (df['Protocol']==protocol)]
        # print(delay, len(new_df))
        sns.boxplot(x=new_df['Condition'], y=new_df['Slope']*new_df['AUC'],ax=ax2[ind])
        fig2.suptitle(f'Slope*AUC for delay of {delay}ms')
        ax2[ind].set_title(protocol)
        # fig2.savefig(savedir+f'\BoxPlot_SlopeAUC_{delay}_{protocol}.png')
        # plt.close()
###############################################################################
###############################################################################
'''
