# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:54:52 2021

@author: F.LARENO-FACCINI
"""
import numpy as np
import extrapy.Organize as og
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def stim_vs_control(df,what,delays,protocols,savedir):
    for delay in delays:
        for protocol in protocols:
            ns_df = df.loc[(df['Delay']==delay)&(df['Condition']=='No Stim')&(df['Protocol']==protocol)]
            s_df = df.loc[(df['Delay']==delay)&(df['Condition']=='Stim')&(df['Protocol']==protocol)]
            females = (6409, 173, 174, 176, 6934, 6456, 6457)
            W,p_value = stats.ttest_rel(ns_df[what],s_df[what],alternative='greater')
            
            effectsize = abs((np.mean(ns_df[what])-np.mean(s_df[what]))/np.mean([np.std(ns_df[what]),np.std(s_df[what])]))
            print('Effect Size', protocol, delay,effectsize)
            
            # print(protocol, delay, p_value)
            if p_value < 0.05:
                print(protocol, delay, p_value)
                
            plt.figure()
            sns.boxplot(x=ns_df.append(s_df)['Condition'], y=ns_df.append(s_df)[what])
            plt.title(f'{what} {delay}, {protocol} (p-value: {p_value})')
            plt.savefig(savedir+f'/Box_{delay}_{protocol}_{what}.pdf')

            # Paired Plot
            plt.figure(figsize=(7,15))
            plt.title(f'{what} {protocol}, {delay} (p-value: {p_value})')
            # Plot the points of the individual mice 
            plt.scatter(np.zeros(len(ns_df[what])), ns_df[what], color='k', alpha=0.6)
            plt.scatter(np.ones(len(s_df[what])), s_df[what], color='k', alpha=0.6)
            # Plot the points of the median
            plt.scatter(0, np.median(list(ns_df[what])), color='r')
            plt.scatter(1, np.median(list(s_df[what])), color='r')
            # Plot the lines of the median
            plt.plot([0, 1], [np.median(list(ns_df[what])),
                              np.median(list(s_df[what]))], c='r')
            # Plot the lines of the individual mice
            for i in range(len(ns_df[what])):
                # Differentiating between males and female (currently the color is the same since I don't care for the difference)
                if list(ns_df['Mouse'])[i] not in females:
                    plt.plot([0, 1], [list(ns_df[what])[i],
                                      list(s_df[what])[i]], c='k', alpha=0.6)
                else:
                    plt.plot([0, 1], [list(ns_df[what])[i],
                                      list(s_df[what])[i]], c='k', alpha=0.6)
                plt.text(-0.1, list(ns_df[what])[i],
                         f"{int(list(ns_df['Mouse'])[i])}")
                
            plt.text(0, np.median(list(ns_df[what])), 'Median')
            plt.xticks([0, 1], ['Control', 'Photostim.'])
            plt.xlim(-0.15, 1.1)
                
                # plt.savefig(savedir+f'/Paired_{delay}_{protocol}_{what}.pdf')


def stim_vs_control_sex(df,sexes,what,delays,protocols,savedir):
    for delay in delays:
        for protocol in protocols:
            fig1,ax1 = plt.subplots(1,2,sharey=True)
            fig2,ax2 = plt.subplots(1,2,sharey=True)
            
            for ind,sex in enumerate(sexes):
                ns_df = df.loc[(df['Delay']==delay)&(df['Condition']=='No Stim')&(df['Protocol']==protocol)&(df['Sex']==sex)]
                s_df = df.loc[(df['Delay']==delay)&(df['Condition']=='Stim')&(df['Protocol']==protocol)&(df['Sex']==sex)]
    
                W,p_value = stats.ttest_rel(ns_df[what],s_df[what],alternative='greater')
                effectsize = abs((np.mean(ns_df[what])-np.mean(s_df[what]))/np.mean([np.std(ns_df[what]),np.std(s_df[what])]))
                print('Effect Size',sex, protocol, delay,effectsize)
                if p_value < 0.05:
                    print(protocol, delay, sex, p_value)
                
                sns.boxplot(x=ns_df.append(s_df)['Condition'], y=ns_df.append(s_df)[what],ax=ax1[ind])
                fig1.suptitle(f'{what} {delay}, {protocol}')
                ax1[ind].set_title(sex)

                # Paired Plot
                fig2.suptitle(f'{what} {protocol}, {delay}')
                # Plot the points of the individual mice 
                ax2[ind].scatter(np.zeros(len(ns_df[what])), ns_df[what], color='k', alpha=0.6)
                ax2[ind].scatter(np.ones(len(s_df[what])), s_df[what], color='k', alpha=0.6)
                # Plot the points of the median
                ax2[ind].scatter(0, np.median(list(ns_df[what])), color='r')
                ax2[ind].scatter(1, np.median(list(s_df[what])), color='r')
                # Plot the lines of the median
                ax2[ind].plot([0, 1], [np.median(list(ns_df[what])),
                                  np.median(list(s_df[what]))], c='r')
                # Plot the lines of the individual mice
                for i in range(len(ns_df[what])):
                    ax2[ind].plot([0, 1], [list(ns_df[what])[i],
                                          list(s_df[what])[i]], c='k', alpha=0.6)
                    ax2[ind].text(-0.1, list(ns_df[what])[i],
                             f"{int(list(ns_df['Mouse'])[i])}")
                    
                ax2[ind].text(0, np.median(list(ns_df[what])), 'Median')
                ax2[ind].set_xticks([0, 1])
                ax2[ind].set_xticklabels(['Control', 'Photostim.'])
                ax2[ind].set_xlim(-0.15, 1.1)
                ax2[ind].set_title(sex)
            # fig2.savefig(savedir+f'/by sex/Paired_{delay}_{protocol}_{what}.pdf')
            # fig1.savefig(savedir+f'/by sex/Box_{delay}_{protocol}_{what}.pdf')

    
def sex_control(df,what,delays,protocols,savedir):
    for delay in delays:
        for protocol in protocols:
            f_df = df.loc[(df['Delay']==delay)&(df['Condition']=='No Stim')&(df['Protocol']==protocol)&(df['Sex']=='Female')]
            m_df = df.loc[(df['Delay']==delay)&(df['Condition']=='No Stim')&(df['Protocol']==protocol)&(df['Sex']=='Male')]

            W,p_value = stats.mannwhitneyu(f_df[what],m_df[what])#,alternative='greater')
            # if p_value < 0.05:
            print(protocol, delay, p_value)
            
            plt.figure()
            sns.boxplot(x=f_df.append(m_df)['Sex'], y=f_df.append(m_df)[what])
            plt.title(f'{what} {delay}, {protocol} (p-value: {p_value})')
            # plt.savefig(savedir+f'/Box_{delay}_{protocol}_{what}.pdf')
            

# =============================================================================
# EXPERIMENTAL DAYS
# =============================================================================
path = r'D:/F.LARENO.FACCINI/RESULTS/New Results/Behaviour/Anticipation_LinReg/Anticipation_df'
df = og.pickle_loading(path)

savedir = r'D:/F.LARENO.FACCINI/RESULTS/New Results/Behaviour/Anticipation_LinReg/Significant Paired Plot'

delays = ('Fixed Delay', 'Random Delay')
sexes = ('Male','Female')
protocols = ('P13', 'P15' ,'P16', 'P18')
females = (6409, 173, 174, 176, 6934, 6456, 6457)


stim_vs_control(df=df,what='AUC',delays=delays,protocols=protocols,savedir=savedir)

# stim_vs_control_sex(df=df,what='Slope',sexes=sexes,delays=delays,protocols=protocols,savedir=savedir)

# sex_control(df=df,what='Intercept',delays=delays,protocols=protocols,savedir=savedir)

# =============================================================================
# TRAINING (abandoned, inconclusive)
# =============================================================================

# path = r'D:/F.LARENO.FACCINI/RESULTS/Behaviour/Anticipation_LinReg/Anticipation_Training_df'
# df = og.pickle_loading(path)

# # Linear Regression
# fig,ax = plt.subplots(1,2,sharey=True)

# for i,sex in enumerate(sexes):
#     new_df = df.loc[(df['Sex']==sex)]
#     X = np.asarray([int(a) for a in new_df['Session']])#.reshape(-1,1)
#     Y = new_df['Slope'].values#.reshape(-1, 1)
    
#     pearson = stats.pearsonr(X, Y)
#     print(pearson)
    
#     linear_regressor = LinearRegression()  # create object for the class
#     linear_regressor.fit(X, Y)  # perform linear regression
#     Y_pred = linear_regressor.predict(X)  # make predictions
#     r2=linear_regressor.score(X,Y)
#     ax[i].scatter(X, Y)
#     ax[i].plot(X, Y_pred, color='red')
#     ax[i].set_title('{} (r2: {:3f})'.format(sex,r2))
#     ax[i].set_xlabel('Session (from the end of the training)')
# ax[0].set_ylabel('Slope of licks before the reward')
# plt.show()


# polynomial regression

# from sklearn.metrics import r2_score

# fig,ax = plt.subplots(1,2,sharey=True)
# for i,sex in enumerate(sexes):
#     new_df = df.loc[(df['Sex']==sex)]
#     X = np.asarray([int(a) for a in new_df['Session']])
#     Y = new_df['Slope'].values
    
#     mymodel = np.poly1d(np.polyfit(X, Y, 3))
#     r2_poly = r2_score(Y, mymodel(X))
#     myline = np.linspace(-6, -1, 100)
    
#     ax[i].scatter(X, Y)
#     ax[i].plot(myline, mymodel(myline),color='r')
#     ax[i].set_title('{} (r2: {:3f})'.format(sex,r2_poly))
#     ax[i].set_xlabel('Session (from the end of the training)')
# ax[0].set_ylabel('Slope of licks before the reward')
# plt.show()



