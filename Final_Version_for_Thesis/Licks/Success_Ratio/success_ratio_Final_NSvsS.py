# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:07:55 2021

@author: F.LARENO-FACCINI
"""
import extrapy.Organize as og
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation as mad
import scipy.stats as stats
import seaborn as sns


def ratios_licks(mice, females):
    delay_list = ['Fixed Delay','Random Delay']
    list_ratios = []
    for group, topi in mice.items():
        for mouse in topi:
            # print(mouse)
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
                
                for key, value in protocols.items():
                    basedir = fr'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\Behaviour\Group {group}\{mouse} (CM16-Buz - {gender})\{delay}\{key}'
                    lick_path = basedir+'\\' + og.file_list(basedir, no_extension=False, ext='.lick')[0]
                    licks = og.remove_empty_trials(lick_path, skip_last=skip_last, end_time='reward')


                    for cond in value:
                        print(key, cond)
                            
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

                        print(delay, len_session)
                        success_ratio = len(good_trials)/len_session
                        success = len(good_trials)
                        
                        if success_ratio>100:
                            print(f'{mouse} {key} {cond} {delay} has a success of {success_ratio}...something is wrong!')
                        
                        temp__ = [success_ratio,success,mouse,gender,delay, f'{key}_{cond}'] # build the dataframe
                        list_ratios.append(temp__)

    df = pd.DataFrame(list_ratios,columns = ('Success_Ratio', 'Success', 'Mouse', 'Sex', 'Delay', 'Condition')) # dataframe columns names

    return df





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
        for protocol in protocols:
            ns_df = df.loc[(df['Delay']==delay)&(df['Condition']==f'{protocol}_No Stim')]
            s_df = df.loc[(df['Delay']==delay)&(df['Condition']==f'{protocol}_Stim')]
            W,p_value = stats.ttest_rel(ns_df['Success_Ratio'],s_df['Success_Ratio'],alternative=alternative)
            
            effectsize = abs((np.mean(ns_df['Success_Ratio'])-np.mean(s_df['Success_Ratio']))/np.mean([np.std(ns_df['Success_Ratio']),np.std(s_df['Success_Ratio'])]))
            print('ES', protocol, delay,effectsize)
            print('pval', protocol, delay, p_value)


            # print(protocol,delay)
            # print('NoStim', ns_df['Success_Ratio'].mean(), ns_df['Success_Ratio'].std())
            # print('Stim', s_df['Success_Ratio'].mean(), s_df['Success_Ratio'].std())


            # if p_value < 0.05:
            #     print('pval', protocol, delay, p_value)
            # Box Plot for the selected pair
            
            # # Check Normality
            # _,pval1 = stats.shapiro(ns_df['Success_Ratio'])
            # _,pval2 = stats.shapiro(s_df['Success_Ratio'])
            # print(delay, protocol, 'ns', pval1)
            # print(delay, protocol, 's', pval2)
            
            
            plt.figure()
            sns.boxplot(x=ns_df.append(s_df)['Condition'], y=ns_df.append(s_df)['Success_Ratio'])
            plt.title(f'{protocol}, {delay} (p-value: {p_value})')
            if save:
                plt.savefig(savedir+fr'\BoxPlot_{protocol}_{delay}.pdf')
                plt.close()

            # Paired Plot
            plt.figure(figsize=(7,15))
            plt.title(f'{protocol}, {delay} (p-value: {p_value})')
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
                plt.savefig(savedir+fr'\PairedPlot_{protocol}_{delay}.pdf')
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
                effectsize =  og.effectsize(ns_df['Success_Ratio'], s_df['Success_Ratio'])
                print('ES', protocol, delay,effectsize)

                
                if p_value < 0.05:
                    print(protocol, delay, sex, p_value)
                # Box Plot for the selected pair
                sns.boxplot(x=ns_df.append(s_df)['Condition'], y=ns_df.append(s_df)['Success_Ratio'],ax=ax1[ind])
                ax1[ind].set_title('{} (p-value: {:3f})'.format(sex,p_value))
                fig1.suptitle(f'{protocol}, {delay}')
    
                # Paired Plot
                ax2[ind].set_title('{} (p-value: {:3f}, , Effect size: {})'.format(sex,p_value,effectsize))
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



def compare_p0 (df,savedir,alternative='two-sided',save=False):
    new_df = df.loc[(df['Condition']=='P0_No Stim')]
    W,p_value = stats.mannwhitneyu(new_df.Success_Ratio[new_df.Sex == 'Female'],new_df.Success_Ratio[new_df.Sex == 'Female'],alternative=alternative)
    print(p_value)
    # Box Plot for the selected pair
    plt.figure()
    sns.boxplot(x=new_df['Sex'], y=new_df['Success_Ratio'])
    plt.title('Male vs Female succes ratio in control condition (p-value: {:2f})'.format(p_value))
    if save:
        plt.savefig(savedir+'\BoxPlot_p0_MvsF.pdf')
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

# df= ratios_licks(mice,females)

####saving dict via pickle#####
# og.pickle_saving(savedir+'\\Success_Ratio_till_reward', df)


df = og.pickle_loading(savedir+'\\Success_Ratio_till_reward')


# =============================================================================
# TOGETHER
# =============================================================================
# general_plot(df,savedir=savedir,save=False)
# paired(df=df,females=females,protocols=protocols, savedir=savedir+'\\by protocol', alternative='two-sided', save=False)

# =============================================================================
# BY SEX
# =============================================================================
# general_plot_sex(df,savedir=savedir,save=True)
paired_sex(df=df,protocols=protocols, savedir=savedir+'\\by protocol\\by sex', alternative='two-sided', save=False)

# compare_p0(df=df,save=True,savedir=savedir)


