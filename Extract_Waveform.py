# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:44:09 2019
@author: Ludo and Fede (modified from Sam Garcia) + Anna

This script allows to extract spike times for each cluster from tridesclous catalogue
in excel sheet
+ figure plot 
This script should run in a tridesclous environement where TDC is installed 
Take a look at the DATA INFO section, fill the info, run and let the magic happen. 
"""
#------------------------------------------------------------------------------
#-----------------------------DATA INFO----------------------------------------
#----------------------------FILL BELOW----------------------------------------

# Some parameters
depth = 1300
protocol = 'P13'
power = '7mW'
stim = 'NoStim'

#The path of the TDC catalogue file - must be STRING format
path =fr'D:/F.LARENO.FACCINI/Preliminary Results/Ephy/6336 (Atlas - Male)/h5/{depth}/{protocol}/rbf/No stim/tdc_6336_{depth}_{protocol}_{power}_{stim}_SingleUnit/'

#Name of the experiment, protocol... anything to discriminate the date. Will be used
#for datasheets/figure labeling - must be STRING format  
name = f'6336-{depth}-{protocol}-{stim}'

#Where to save datasheets and figures. If None, nothing will be saved  - must be STRING format
savedir = r'C:/Users/F.LARENO-FACCINI/Desktop/wf_prova'

#Specify the channel group to explore as [#]. Feel free to do them all : [0,1,2,3]
channel_groups=[0]

#If True : close figure automatically (avoids to overhelm screen when looping and debug)
closefig = True

#The opacity for the waveform rms. 1. = solid, 0. = transparent 
wf_alpha =0.2


#------------------------------------------------------------------------------
#-----------------------------THE SCRIPT---------------------------------------
#---------------------------DO NOT MODIFY--------------------------------------
import tridesclous as tdc 
import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd 
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

#Load the catalogue
dataio = tdc.DataIO(path)


for chan_grp in channel_groups:
    
    #Define the constructor and the channel group 
    cc = tdc.CatalogueConstructor(dataio, chan_grp=chan_grp)
    print ('--- Experiment : {} ---'.format(name))
    print ('Catalogue loaded from {}'.format(path))
    print ('----Channel group {}----'.format(chan_grp))    
    
    #The cluster label for median waveform, the median waveforms and the median rms
    waveforms_label = cc.clusters['cluster_label']
    waveforms = cc.centroids_median
    wf_rms =cc.clusters['waveform_rms']
    
    #The probe geometry and specs 
    probe_geometry = cc.geometry

    #Positive clusters list
    clust_list = np.unique(waveforms_label)
    mask=(clust_list >= 0)
    pos_clustlist=clust_list[mask]
    n_clust=len(pos_clustlist)    


    # Figure for waveforms------------------------------------------------------
          
    for cluster, idx in zip(pos_clustlist, range(n_clust)):
        fig1 = plt.figure(figsize=(10,12))
        ax1 = fig1.add_subplot(111)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        fig1.suptitle('{} Average Waveform Cluster {} (ch_group = {})'.format(name,cluster,chan_grp))
        ax1.set_xlabel('Probe location (micrometers)')
        ax1.set_ylabel('Probe location (micrometers)')
        fig2.suptitle(f'{name} - Peak Amp (of avg waveform) by channel for Cluster {cluster}')
        ax2.set_xlabel('Channel')
        ax2.set_ylabel('Peak Amplitude (Z score)')


        for loc, prob_loc in zip(range(len(probe_geometry)), probe_geometry): # N.B. the index of the channels is the real number of the probe site! (the channels at index [0] IS site #0 on the probe)
            x_offset, y_offset = prob_loc[0], prob_loc[1]
            #base_x = np.arange(0,len(waveforms[1,:,loc]),1)  
            base_x = np.linspace(-15,15,num=len(waveforms[idx+1,:,loc])) #Basic x-array for plot, centered
            clust_color = 'C{}'.format(idx)

            wave = waveforms[idx+1,:,loc]+y_offset
            ax1.plot(base_x+2*x_offset,wave,color=clust_color)
            ax1.fill_between(base_x+2*x_offset,wave-wf_rms[idx+1],wave+wf_rms[idx+1], color=clust_color,alpha=wf_alpha)

        for i in range(16):
            ax2.plot(np.arange(0,16,1),waveforms[idx+1].min(axis=0))
        
        
        if savedir !=None :
            fig1.savefig('{}/{}_Waveforms_changrp{}_Cluster{}.pdf'.format(savedir,name,chan_grp,cluster))         
            fig2.savefig(f'{savedir}/{name}_PeakAmp_Waveforms_changrp{chan_grp}_Cluster{cluster}.pdf')         
            with pd.ExcelWriter('{}/{}_waveforms_changrp{}.xlsx'.format(savedir,name,chan_grp)) as writer:
                #File infos 
                waveform_info = pd.DataFrame(cc.clusters)
                waveform_info.to_excel(writer, sheet_name='info')
                for cluster, idx in zip(pos_clustlist, range(n_clust)):
                    clust_WF = pd.DataFrame(waveforms[idx+1,:,:])      
                    clust_WF.to_excel(writer,sheet_name='cluster {}'.format(cluster))           
        else : 
            print ('No savedir specified : nothing will be saved')
        
        if closefig==True:
            plt.close('all')