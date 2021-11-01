# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 09:27:17 2021

@author: F.LARENO-FACCINI
"""

import extrapy.Organize as og
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import scikit_posthocs as sp
import pingouin as pg


def ratios_licks(mice, females):
    delay_list = ['Fixed Delay','Random Delay']
    list_ratios = []
    for group, topi in mice.items():
        for mouse in topi:
            print(mouse)
            protocols = {'P0': ['No Stim'],
                         'P13': ['No Stim', 'Stim'],
                         'P15': ['No Stim', 'Stim'],
                         'P16': ['No Stim', 'Stim'],
                         'P18': ['No Stim', 'Stim']}
    
            gender = 'Female' if mouse in females else 'Male'
    
            if int(group) < 15:
                del protocols['P0']
    
            for delay in delay_list:
                skip_last = True if delay == 'Random Delay' else False
                
                all_good_ns,all_len_ns = [], []        
                all_good_s,all_len_s = [], [] 
    
                for key, value in protocols.items():
                    basedir = fr'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\Behaviour\Group {group}\{mouse} (CM16-Buz - {gender})\{delay}\{key}'
                    lick_path = basedir+'\\' + og.file_list(basedir, no_extension=False, ext='.lick')[0]
                    licks,*_ = og.remove_empty_trials(lick_path, skip_last=skip_last, end_time='reward')
    
    
                    for cond in value:
                        # print(key, cond)
                            
                        good_trials = np.unique(licks[:, 0])
    
                        
                        if cond == 'No Stim' and 'P1' in key:
                            good_trials = good_trials[good_trials < 31]
                            len_session = 0.3
                        
                        elif cond == 'Stim':
                            good_trials = good_trials[good_trials > 30]
                            if delay == 'Random Delay':
                                len_session = 0.29
                            else:
                                len_session = 0.3
                            
                        else:
                            len_session = 0.6
                            if delay == 'Random Delay':
                                len_session = 0.59
                        
                        if cond == 'No Stim':
                            all_good_ns.append(len(good_trials))
                            all_len_ns.append(len_session)
                        else:
                            all_good_s.append(len(good_trials))
                            all_len_s.append(len_session)
    
                all_good_ns = np.sum(np.array(all_good_ns))
                all_len_ns = np.sum(np.array(all_len_ns))
                success_ratio = all_good_ns/all_len_ns
                success = all_good_ns
                temp__ = [success_ratio,success,mouse,gender,delay, 'No Stim'] # build the dataframe
                list_ratios.append(temp__)
    
                all_good_s = np.sum(np.array(all_good_s))
                all_len_s = np.sum(np.array(all_len_s))
                success_ratio = all_good_s/all_len_s
                success = all_good_s            
                temp__ = [success_ratio,success,mouse,gender,delay, 'Stim'] # build the dataframe
                list_ratios.append(temp__)
    
    return pd.DataFrame(list_ratios,columns = ('Success_Ratio', 'Success', 'Mouse', 'Sex', 'Delay', 'Condition')) # dataframe columns names



def general_plot(df, savedir=None, save=False, violin=False, box=True):
    delay_list = ['Fixed Delay','Random Delay']
    for delay in delay_list:
        new_df = df.loc[(df['Delay']==delay)]
        # print(delay, len(new_df))
        if violin:
            # Violin Plot
            plt.figure(figsize=(15,7))
            sns.violinplot(x=new_df['Condition'], y=new_df['Success_Ratio'])
            plt.title(f'Success Ratio for delay of {delay}')
            if save:
                plt.savefig(savedir+f'\Violin_Plot_{delay}.pdf')
                plt.close()
        if box:
            # Box Plot
            plt.figure(figsize=(15,7))
            sns.boxplot(x=new_df['Condition'], y=new_df['Success_Ratio'])
            plt.title(f'Success Ratio for {delay}')
            if save:
                plt.savefig(savedir+f'\BoxPlot_{delay}.pdf')
                plt.close()


def general_plot_sex(df, savedir=None, save=False, violin=False, box=True):
    delay_list = ['Fixed Delay','Random Delay']
    sexes = ['Female','Male']
    
    for delay in delay_list:
        fig,ax = plt.subplots(2,1,sharex=True,sharey=True, figsize=(15,12))
        
        for ind,sex in enumerate(sexes):
            new_df = df.loc[(df['Delay']==delay)&(df['Sex']==sex)]

            sns.boxplot(x=new_df['Condition'], y=new_df['Success_Ratio'],ax=ax[ind])
            ax[ind].set_title(sex)
            ax[ind].set_ylabel('Success Ratio (%)')
            
        fig.suptitle(f'Success Ratio for {delay}')
        if save:
            plt.savefig(savedir+f'\BoxPlot_{delay}_by_sex.pdf')
            plt.close()


def paired(df, females, protocols, savedir, alternative='two-sided', save=False):
    delay_list = ['Fixed Delay','Random Delay']
    for delay in delay_list:
            ns_df = df.loc[(df['Delay']==delay)&(df['Condition']=='No Stim')]
            s_df = df.loc[(df['Delay']==delay)&(df['Condition']=='Stim')]
            # print('NS', stats.shapiro(ns_df['Success_Ratio']))
            # print('S', stats.shapiro(s_df['Success_Ratio']))
            W,p_value = stats.ttest_rel(ns_df['Success_Ratio'],s_df['Success_Ratio'],alternative=alternative)
            effect = og.effectsize(ns_df['Success_Ratio'], s_df['Success_Ratio'])

            if p_value < 0.05:
                print(delay, p_value)
                print(effect)
            # Box Plot for the selected pair
            
            # # Check Normality
            # _,pval1 = stats.shapiro(ns_df['Success_Ratio'])
            # _,pval2 = stats.shapiro(s_df['Success_Ratio'])
            # print(delay, protocol, 'ns', pval1)
            # print(delay, protocol, 's', pval2)
            
            
            plt.figure()
            sns.boxplot(x=ns_df.append(s_df)['Condition'], y=ns_df.append(s_df)['Success_Ratio'])
            plt.title(f'{delay} (p-value: {p_value})')
            if save:
                plt.savefig(savedir+fr'\BoxPlot_{delay}.pdf')
                plt.close()

            # Paired Plot
            plt.figure(figsize=(7,15))
            plt.title(f'{delay} (p-value: {p_value})')
            # Plot the points of the individual mice 
            plt.scatter(np.zeros(len(ns_df['Success_Ratio'])), ns_df['Success_Ratio'], color='k', alpha=0.6)
            plt.scatter(np.ones(len(s_df['Success_Ratio'])), s_df['Success_Ratio'], color='k', alpha=0.6)
            # Plot the points of the median
            plt.scatter(0, np.median(list(ns_df['Success_Ratio'])), color='r')
            plt.scatter(1, np.median(list(s_df['Success_Ratio'])), color='r')
            # Plot the lines of the median
            plt.plot([0, 1], [np.median(list(ns_df['Success_Ratio'])),
                              np.median(list(s_df['Success_Ratio']))], c='r')
            # Plot the lines of the individual mice
            for i in range(len(ns_df['Success_Ratio'])):
                # Differentiating between males and female (currently the color is the same since I don't care for the difference)
                if list(ns_df['Mouse'])[i] not in females:
                    plt.plot([0, 1], [list(ns_df['Success_Ratio'])[i],
                                      list(s_df['Success_Ratio'])[i]], c='k', alpha=0.6)
                else:
                    plt.plot([0, 1], [list(ns_df['Success_Ratio'])[i],
                                      list(s_df['Success_Ratio'])[i]], c='k', alpha=0.6)
                plt.text(-0.1, list(ns_df['Success_Ratio'])[i],
                         f"{int(list(ns_df['Mouse'])[i])}")
                
            plt.text(0, np.median(list(ns_df['Success_Ratio'])), 'Median')
            plt.xticks([0, 1], ['Control', 'Photostim.'])
            plt.xlim(-0.15, 1.1)
            if save:
                plt.savefig(savedir+fr'\PairedPlot_{delay}.pdf')
                plt.close()
    

def paired_sex(df, protocols, savedir, alternative='two-sided', save=False):
    delay_list = ['Fixed Delay','Random Delay']
    sexes = ['Female','Male']
    
    for delay in delay_list:
        for protocol in protocols:
            fig1,ax1 = plt.subplots(1,2,sharey=True)
            fig2, ax2 = plt.subplots(1,2,sharey=True)
            
            for ind,sex in enumerate(sexes):
                ns_df = df.loc[(df['Delay']==delay)&(df['Condition']==f'{protocol}_No Stim')&(df['Sex']==sex)]
                s_df = df.loc[(df['Delay']==delay)&(df['Condition']==f'{protocol}_Stim')&(df['Sex']==sex)]
                W,p_value = stats.ttest_rel(ns_df['Success_Ratio'],s_df['Success_Ratio'],alternative=alternative)
                effect = og.effectsize(ns_df['Success_Ratio'], s_df['Success_Ratio'])
                print(effect)
                
                if p_value < 0.05:
                    print(protocol, delay, sex, p_value)
                # Box Plot for the selected pair
                sns.boxplot(x=ns_df.append(s_df)['Condition'], y=ns_df.append(s_df)['Success_Ratio'],ax=ax1[ind])
                ax1[ind].set_title('{} (p-value: {:3f})'.format(sex,p_value))
                fig1.suptitle(f'{protocol}, {delay}')
    
                # Paired Plot
                ax2[ind].set_title('{} (p-value: {:3f})'.format(sex,p_value))
                fig2.suptitle(f'{protocol}, {delay}')
                
                # Plot the points of the individual mice 
                ax2[ind].scatter(np.zeros(len(ns_df['Success_Ratio'])), ns_df['Success_Ratio'], color='k', alpha=0.6)
                ax2[ind].scatter(np.ones(len(s_df['Success_Ratio'])), s_df['Success_Ratio'], color='k', alpha=0.6)
               
                # Plot the points of the median
                ax2[ind].scatter(0, np.median(list(ns_df['Success_Ratio'])), color='r')
                ax2[ind].scatter(1, np.median(list(s_df['Success_Ratio'])), color='r')
                
                # Plot the lines of the median
                ax2[ind].plot([0, 1], [np.median(list(ns_df['Success_Ratio'])),
                                  np.median(list(s_df['Success_Ratio']))], c='r')
                
                # Plot the lines of the individual mice
                for i in range(len(ns_df['Success_Ratio'])):
                    # Differentiating between males and female (currently the color is the same since I don't care for the difference)
                    ax2[ind].plot([0, 1], [list(ns_df['Success_Ratio'])[i],
                                          list(s_df['Success_Ratio'])[i]], c='k', alpha=0.6)
                    ax2[ind].text(-0.1, list(ns_df['Success_Ratio'])[i],
                             f"{int(list(ns_df['Mouse'])[i])}")
                    
                ax2[ind].text(0, np.median(list(ns_df['Success_Ratio'])), 'Median')
                ax2[ind].set_xticks([0, 1], minor=False)
                ax2[ind].set_xticklabels(['Control', 'Photostim.'])
                ax2[ind].set_xlim(-0.15, 1.1)
            if save:
                fig2.savefig(savedir+f'\PairedPlot_{protocol}_{delay}.pdf')
                fig1.savefig(savedir+f'\BoxPlot_{protocol}_{delay}.pdf')
                plt.close()

def paired_foreperiod(df, females, protocols, savedir, alternative='two-sided', save=False):
    conditions = ['No Stim','Stim']

    for cond in conditions:
            ns_df = df.loc[(df['Delay']=='Fixed Delay')&(df['Condition']==cond)]
            s_df = df.loc[(df['Delay']=='Random Delay')&(df['Condition']==cond)]

            print(ns_df.describe())
            print(s_df.describe())
            # print('NS', stats.shapiro(ns_df['Success_Ratio']))
            # print('S', stats.shapiro(s_df['Success_Ratio']))
            W,p_value = stats.ttest_rel(ns_df['Success_Ratio'],s_df['Success_Ratio'],alternative=alternative)
            effect = og.effectsize(ns_df['Success_Ratio'], s_df['Success_Ratio'])
            if p_value < 0.05:
                print(cond, p_value)
                print(effect)
            # Box Plot for the selected pair
            
            # # Check Normality
            # _,pval1 = stats.shapiro(ns_df['Success_Ratio'])
            # _,pval2 = stats.shapiro(s_df['Success_Ratio'])
            # print(delay, protocol, 'ns', pval1)
            # print(delay, protocol, 's', pval2)
            
            
            plt.figure()
            sns.boxplot(x=ns_df.append(s_df)['Delay'], y=ns_df.append(s_df)['Success_Ratio'])
            plt.title(f'{cond} (p-value: {p_value}, Effect size: {effect})')
            if save:
                plt.savefig(savedir+fr'\FvsR_BoxPlot_{cond}.pdf')
                plt.close()

            # Paired Plot
            plt.figure(figsize=(7,15))
            plt.title(f'{cond} (p-value: {p_value})')
            # Plot the points of the individual mice 
            plt.scatter(np.zeros(len(ns_df['Success_Ratio'])), ns_df['Success_Ratio'], color='k', alpha=0.6)
            plt.scatter(np.ones(len(s_df['Success_Ratio'])), s_df['Success_Ratio'], color='k', alpha=0.6)
            # Plot the points of the median
            plt.scatter(0, np.median(list(ns_df['Success_Ratio'])), color='r')
            plt.scatter(1, np.median(list(s_df['Success_Ratio'])), color='r')
            # Plot the lines of the median
            plt.plot([0, 1], [np.median(list(ns_df['Success_Ratio'])),
                              np.median(list(s_df['Success_Ratio']))], c='r')
            # Plot the lines of the individual mice
            for i in range(len(ns_df['Success_Ratio'])):
                # Differentiating between males and female (currently the color is the same since I don't care for the difference)
                if list(ns_df['Mouse'])[i] not in females:
                    plt.plot([0, 1], [list(ns_df['Success_Ratio'])[i],
                                      list(s_df['Success_Ratio'])[i]], c='k', alpha=0.6)
                else:
                    plt.plot([0, 1], [list(ns_df['Success_Ratio'])[i],
                                      list(s_df['Success_Ratio'])[i]], c='k', alpha=0.6)
                plt.text(-0.1, list(ns_df['Success_Ratio'])[i],
                         f"{int(list(ns_df['Mouse'])[i])}")
                
            plt.text(0, np.median(list(ns_df['Success_Ratio'])), 'Median')
            plt.xticks([0, 1], ['Fixed', 'Random.'])
            plt.xlim(-0.15, 1.1)
            if save:
                plt.savefig(savedir+fr'\FvsR_PairedPlot_{cond}.pdf')
                plt.close()


############################################################################################################################
############################################################################################################################
############################################################################################################################
savedir = r'D:\F.LARENO.FACCINI\RESULTS\New Results\Behaviour\Success Ratio'

mice = {'14': (6401, 6402, 6409),
        '15': (173, 174, 176),
        '16': (6924, 6928, 6934),
        '17': (6456, 6457)}
protocols = ('P13', 'P15' ,'P16', 'P18')

females = (6409, 173, 174, 176, 6934, 6456, 6457)

df= ratios_licks(mice,females)

####saving dict via pickle#####
# og.pickle_saving(savedir+'\\Success_Ratio_till_reward', df)


# df = og.pickle_loading(savedir+'\\Success_Ratio_till_reward')


# =============================================================================
# TOGETHER
# =============================================================================
general_plot(df,savedir=savedir,save=False)
paired(df=df,females=females,protocols=protocols, savedir=savedir+'\\by protocol', alternative='two-sided', save=False)
# paired_foreperiod(df=df,females=females,protocols=protocols, savedir=savedir+'\\by protocol', alternative='two-sided', save=False)
# =============================================================================
# BY SEX
# =============================================================================
# general_plot_sex(df,savedir=savedir,save=False)
# paired_sex(df=df,protocols=protocols, savedir=savedir+'\\by protocol\\by sex', alternative='greater', save=False)

# compare_p0(df=df,save=True,savedir=savedir)



###################################################################################
#########################    STATS STATS STATS   ##################################
###################################################################################

# welch_df = None
# for session in all_values.keys():
#     # if '_NoStim' in session:
#     for x in all_values[session]:
#         # print(type(x))
#         data = pd.DataFrame([float(x),f'{session}'], index=[f'{what}', 'Session']).T
#         if welch_df is None:
#             welch_df = data
#         else:
#             welch_df = pd.concat([welch_df,data])
# welch_df[f'{what}'] = pd.to_numeric(welch_df[f'{what}'])
# print(pg.welch_anova(dv='Success_Ratio', between='Session', data=df))

# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
# formula = 'Success_Ratio ~ Delay + Condition + Delay:Condition'
# model = ols(formula, data=df).fit()
# aov_table = sm.stats.anova_lm(model, typ=2, robust='hc3')
# print(aov_table)
# pairwise_tukeyhsd(model, "machine")

