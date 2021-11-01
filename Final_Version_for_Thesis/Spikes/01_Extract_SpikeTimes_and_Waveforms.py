# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:44:09 2019
@author: Ludo and Fede (modified from Sam Garcia) + Anna

This script allows to extract spike times and waveform from tridesclous catalogue
in excel sheet
+ figure plot 
+ spike train viewer
This script should run in a tridesclous environement where TDC is installed 
Take a look at the DATA INFO section, fill the info, run and let the magic happen. 
"""
#------------------------------------------------------------------------------
#-----------------------------DATA INFO----------------------------------------
#----------------------------FILL BELOW----------------------------------------


mice = {
        # '14': (6401, 6409),
        # '15': (173,),
        '16': (6924,)#,
        # '17': (6456,6457)
    }
females = (6409, 173, 6456, 6457)

delays = ('Fixed',)# 'Random')


for group, topi in mice.items():
    protocols = {#'NB': ['No Stim',],
            # 'P0': ['No Stim',],
            # 'P13': ['No Stim', 'Stim'],
            # 'P15': ['No Stim', 'Stim'],
            # 'P16': ['No Stim', 'Stim'],
            # 'P18': ['No Stim', 'Stim'],
            # 'Steady': ['Stim',],
            'Washout': ['No Stim',]}

    
    if int(group) < 15:
        del protocols['NB']
        del protocols['P0']
        del protocols['Washout']


    for mouse in topi:
        gender = 'Male' if mouse not in females else 'Female'

        for protocol, v in protocols.items():
            for condition in v:
                for experiment in delays:
                    print(group,mouse,gender,protocol,condition,experiment)


                    # if protocol == 'NB' or protocol == 'P0':
                    #     condition = None
                    
                    if experiment == 'Fixed':
                        exp = 'EF'
                    elif experiment == 'Random':
                        exp = 'ER'
                    
                    #The path of the TDC catalogue file - must be STRING format
                    if protocol != 'NB' and protocol != 'P0' and protocol != 'Washout':
                        path = fr'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\Ephy\Group {group}\{mouse} (CM16-Buz - {gender})\{experiment} Delay\{protocol}\{condition}\tdc_{mouse}_{exp}_{protocol}_{condition.replace(" ","")}'
                        #Name given to the files produced
                        name = f'{mouse}-{exp}-{protocol}-{condition.replace(" ","")}'
                    
                    else:
                        path = fr'\\equipe2-nas2\F.LARENO-FACCINI\BACKUP FEDE\Ephy\Group {group}\{mouse} (CM16-Buz - {gender})\{experiment} Delay\{protocol}\{condition}\tdc_{mouse}_{exp}_{protocol}'
                        #Name given to the files produced
                        name = f'{mouse}-{exp}-{protocol}'
                    
                    #Where to save datasheets and figures. If None, nothing will be saved  - must be STRING format
                    savedir = r'D:\F.LARENO.FACCINI\RESULTS\New Results\Spike Sorting'
                    
                    
                    sampling_rate = 20000 #in Hz
                    
                    #Stim time ad stim duration in seconds
                    stim_len = {'P13':0.8, 'P15':1.25, 'P16':1.16, 'P18':1}
                    if protocol != 'NB' and protocol != 'P0' and protocol != 'Washout':
                        stim_time = 1.5
                        stim_duration = stim_len[f'{protocol}']
                        print(stim_duration)
                        

                    water_time = 2.55
                    water_duration = 0.15
                    
                    #Lenght of the single episode (in seconds)
                    ep_len = 9.1
                    
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
                    import openpyxl
                    import os
                    #from ephyviewer import mkQApp, MainViewer, SpikeTrainViewer
                    #from ephyviewer import InMemorySpikeSource
                    from openpyxl.utils.dataframe import dataframe_to_rows
                    
                    
                    # #SPIKE TRAIN VIEWER-----------------------------------------------------------
                    # def spikeviewer(SPIKES):
                    #     all_spikes =[]
                    #     for label in range (len(SPIKES)):     
                    #         spike_name = 'Unit {}#{}'.format(chan_grp, label)
                    #         all_spikes.append({ 'time':SPIKES[label] , 'name':spike_name})    
                    #     spike_source = InMemorySpikeSource(all_spikes=all_spikes)        
                    #     win = MainViewer(debug=False, show_global_xsize=True, show_auto_scale=True)
                    #     view1 = SpikeTrainViewer(source=spike_source)
                    #     win.add_view(view1)     
                    #     return (win)
                    # #-----------------------------------------------------------------------------    
                    
                    # print(path)
                    if os.path.exists(path):
                        #Load the catalogue
                        dataio = tdc.DataIO(path)
                        
                        #Number of segments
                        n_seg=dataio.nb_segment
                        
                        #Compute time vector of the whole trace
                        sampling_period = 1.0/sampling_rate 
                        len_trace=0
                        for seg_num in range(n_seg):  
                            len_seg = dataio.get_segment_length(seg_num)
                            len_trace += len_seg
                        time_vector = np.arange(0,len_trace,1)*sampling_period   
                        
                        #Mean episode lenght 
                        ep_len=(len_trace/n_seg)*sampling_period
                        
                        
                        for chan_grp in channel_groups:
                            
                            #Define the constructor and the channel group 
                            cc = tdc.CatalogueConstructor(dataio, chan_grp=chan_grp)
                            print ('--- Experiment : {} ---'.format(name))
                            print ('Catalogue loaded from {}'.format(path))
                            print ('----Channel group {}----'.format(chan_grp))
                            
                            #The cluster list for the spikes 
                            clust_id = cc.all_peaks['cluster_label']
                            
                            #The spike times
                            spike_index = cc.all_peaks['index']
                            spike_times= spike_index*sampling_period
                            seg_id = cc.all_peaks['segment']   #The segment list for the spikes 
                            for ind in range(len(spike_times)):
                                spike_times[ind]=spike_times[ind]+(ep_len*seg_id[ind])   
                            
                            #Stim vector for the whole trace
                            if protocol != 'NB' and protocol != 'P0' and protocol != 'Washout':
                                stim_vector = np.arange(stim_time,time_vector[-1],float(ep_len))
                            water_vector = np.arange(water_time,time_vector[-1],float(ep_len))
                            
                            #The cluster label for median waveform, the median waveforms and the median rms
                            waveforms_label = cc.clusters['cluster_label']
                            waveforms = cc.centroids_median
                            wf_rms =cc.clusters['waveform_rms']
                            
                            #The probe geometry and specs 
                            probe_geometry = cc.geometry
                            probe_channels = cc.nb_channel
                            
                            #Positive_clusters
                            clust_list=np.unique(waveforms_label)
                            mask=(clust_list >= 0)
                            pos_clustlist=clust_list[mask]
                            n_clust=len(pos_clustlist)
                            delta = len(clust_list)-len(pos_clustlist)
                            
                            #Figure for waveforms------------------------------------------------------
                                  
                            for cluster in pos_clustlist:
                                fig1 = plt.figure(figsize=(10,12))
                                plt.title('{} Average Waveform Cluster {} (ch_group = {})'.format(name,cluster,chan_grp))
                                plt.xlabel('Probe location (micrometers)')
                                plt.ylabel('Probe location (micrometers)')
                                # print(cluster, cluster+delta)
                                
                                for loc, prob_loc in zip(range(len(probe_geometry)), probe_geometry): 
                                    x_offset, y_offset = prob_loc[0], prob_loc[1]
                                    #base_x = np.arange(0,len(waveforms[1,:,loc]),1)  
                                    # print(cluster+delta)
                                    base_x = np.linspace(-15,15,num=len(waveforms[cluster+delta,:,loc])) #Basic x-array for plot, centered
                                    clust_color = 'C{}'.format(cluster)
                        
                                    wave = waveforms[cluster+delta,:,loc]+y_offset
                                    plt.plot(base_x+2*x_offset,wave,color=clust_color)
                                    plt.fill_between(base_x+2*x_offset,wave-wf_rms[cluster+delta],wave+wf_rms[cluster+delta], color=clust_color,alpha=wf_alpha)
                        
                                if savedir !=None :
                                    fig1.savefig(f'{savedir}/Figures/Group {group}/{experiment} Delay/{name}_Waveforms_changrp{chan_grp}_Cluster{cluster}.pdf')         
                            
                            if savedir !=None :
                                # fig1.savefig(f'{savedir}/Figures/Group {group}/{experiment} Delay/{name}_Waveforms_changrp{chan_grp}_Cluster{cluster}.pdf')         
                                with pd.ExcelWriter(f'{savedir}/Waveforms/{experiment} Delay/{name}_waveforms_changrp{chan_grp}.xlsx') as writer:
                                    #File infos 
                                    waveform_info = pd.DataFrame(cc.clusters)
                                    waveform_info.to_excel(writer, sheet_name='info')
                                    for cluster in pos_clustlist:
                                        clust_WF = pd.DataFrame(waveforms[cluster+delta,:,:])      
                                        clust_WF.to_excel(writer,sheet_name=f'cluster {cluster}')           
                            else : 
                                print ('No savedir specified : nothing will be saved')
                            
                            if closefig==True:
                                plt.close()
                                
                        
                            
                            #Spike Times extraction per cluster---------------------------------------- 
                            fig2, ax =plt.subplots(2,1,figsize=(10,5))
                            ax[0].set_title('{} All spike times (ch_group = {})'.format(name,chan_grp))
                            ax[0].eventplot(spike_times, linewidth=0.1)
                            ax[1].set_xlabel('Time (s)')
                            ax[1].set_ylabel('Cluster ID')
                            ticks = np.arange(0,(len_trace/sampling_rate),9)
                            ax[0].set_xticks(ticks)
                            ax[1].set_xticks(ticks)
                            
                            if condition == 'Stim':
                                for stim in stim_vector:
                                    ax[0].axvspan(stim,stim+stim_duration,color='skyblue',alpha=0.6)
                                    ax[1].axvspan(stim,stim+stim_duration,color='skyblue',alpha=0.6)
                        
                            if experiment == 'Fixed':
                                for water in water_vector:
                                    ax[0].axvspan(water,water+water_duration,color='lightcoral',alpha=0.4)
                                    ax[1].axvspan(water,water+water_duration,color='lightcoral',alpha=0.4)
                        
                            SPIKES = [] #To store all the spikes, one array per cluster
                            #cluster_list = [] #To store the cluster for file indexing 
                            
                            for cluster in pos_clustlist:
                                clust_color = 'C{}'.format(cluster)
                                #cluster_list.append(str(cluster))
                                temp_ = [] #To store spikes from each cluster
                                for i,j in enumerate(clust_id):
                                    if j == cluster:
                                        temp_.append(spike_times[i])
                                        
                                SPIKES.append(np.asarray(np.ravel(temp_)))
                                
                                ax[1].eventplot(np.ravel(temp_), lineoffsets=cluster, linelengths=0.5, linewidth=0.5, color=clust_color)
                               
                            #SAVE THE SPIKE DATA (or not) ---------------------------------------------    
                            if savedir != None:
                                sorted_spikes = pd.DataFrame(SPIKES,index=pos_clustlist)
                                sorted_spikes.index.name = 'Cluster' # Index label
                                sorted_spikes = sorted_spikes.transpose()
                                wb = openpyxl.Workbook()
                                ws1 = wb.active
                                ws1.title = 'All Clusters'
                                
                                for r in dataframe_to_rows(sorted_spikes, index=False, header=True):
                                    ws1.append(r)
                             
                            if closefig==True:
                                plt.close()
                               
                                
                            #Spike Times extraction per cluster, aligned segments----------------------------------------     
                            fig3, ax =plt.subplots(2, n_clust, figsize=(20,10), sharex=True, squeeze=False)
                            spike_times= spike_index*sampling_period
                            
                            # ax = ax.ravel()
                            
                            for idx in pos_clustlist:
                                ax[0,idx].set_title('Cluster{}'.format(idx))
                                ax[1,idx].set_xlabel('Time (s)')
                                ax[0,idx].set_ylabel('Segment')        
                                ticks = np.arange(0,ep_len,1)
                                ax[0,idx].set_xticks(ticks)
                                
                                if experiment == 'Fixed':
                                    ax[0,idx].axvspan(water_time, water_time+water_duration,color='lightcoral',alpha=0.4)
                                    ax[1,idx].axvspan(water_time, water_time+water_duration,color='lightcoral',alpha=0.4)
                        
                                if condition == 'Stim':
                                    ax[0,idx].axvspan(stim_time,stim_time+stim_duration,color='skyblue',alpha=0.6)        
                                    ax[1,idx].axvspan(stim_time,stim_time+stim_duration,color='skyblue',alpha=0.6)        
                        
                                clust_color = 'C{}'.format(idx)
                                clust_dict = {}
                                for seg_num in range(n_seg):  
                                    temp_ = [] #To store spikes from each cluster            
                                    for i,j in enumerate(seg_id):
                                        if j == seg_num:
                                            if clust_id[i]==idx:
                                                  temp_.append(spike_times[i])
                                                  
                                    clust_dict[f'Seg_{seg_num}'] = temp_
                                    ax[0,idx].eventplot(np.ravel(temp_), lineoffsets=seg_num, linelengths=0.5, linewidth=0.5, color=clust_color)       
                        
                                df_clust = pd.DataFrame(dict([ (k,pd.Series(v,dtype=float)) for k,v in clust_dict.items() ]))
                                if savedir != None:
                                    ws2 = wb.create_sheet(title=f'Cluster {idx}')
                                    for r in dataframe_to_rows(df_clust, index=False, header=True):
                                        ws2.append(r)
                                
                                
                                # PSTH
                                all_trials = df_clust.to_numpy().ravel()
                                all_trials = np.sort(all_trials[np.isfinite(all_trials)])
                                ax[1,idx].hist(all_trials, bins=90, color=clust_color, linewidth=1.2)       
                                
                                fig3.suptitle(name + ' All cluster aligned')      
                                
                            file_name = f'{savedir}/Spike Times/{experiment} Delay/{name}_Spike_times_changrp{chan_grp}.xlsx'
                            wb.save(file_name)
                        
                            if savedir != None:
                                fig3.savefig(f'{savedir}/Figures/Group {group}/{experiment} Delay/{name}_Spike_times_aligned_changrp{chan_grp}.pdf')
                                
                                
                            if closefig==True:
                                plt.close('all')
                         
                            
                        #Activate spike train viewer--------------------------------------------------
                            # app = mkQApp()
                            # win = spikeviewer(SPIKES)
                            # win.show()
                            # app.exec_()
                    else:
                        print('TDC file inexistent')
