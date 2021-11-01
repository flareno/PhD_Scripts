# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:21:58 2021

@author: F.LARENO-FACCINI
"""

def PrinCompAn(df_pca, threshold=0.95):
    '''
    Computes a Pincipal Component Analysis on a dataframe and returns the explained variance
    of each principal component (PC) and the eigenvalues of each individual.
    df_pca: dataset after using pandas.DataFrame.
        /!\ INDIVIDUALS AS ROWS AND VARIABLES AS COLUMNS
    threshold (float): set the number of principal components t. Default: 0.95.
    Returns a figure with:
        - elbow plot (ax1): gives the number of PCs explaining at least 95% of the total variance.
        - bar plot (ax2): explained variance by each PC.
        - scatter plot (ax3): position of each individual in the new dimentional subspace.
        - bar plot (ax4 and ax5): contribution of each variable on the first (ax4) and second (ax5) PCs.
    '''
    scaler = StandardScaler()
    center_norm_matrix = scaler.fit_transform(df_pca)

    pca = PCA()
    pca_fit = pca.fit(center_norm_matrix)
    ExpVar = pca_fit.explained_variance_ratio_
    cumsum = np.cumsum(ExpVar)
    d = np.argmax(cumsum >= threshold) + 1
    VarComp = pd.DataFrame(pca_fit.components_[:, :d]) #Contribution of each variable on each PC
    PrCoAn = PCA(n_components = d)
    eigenvalue = PrCoAn.fit_transform(center_norm_matrix)
    df_eigen = pd.DataFrame(eigenvalue,
                            columns = [f'PC{i+1}' for i in range(d)],
                            index = df_pca.index)
    x = np.arange(1, len(ExpVar) + 1)
    fig_PCA, ax_PCA = plt.subplots(1, 3, figsize = (17,4), tight_layout = True)
    ax_PCA[0].plot(x, cumsum, marker='o', linestyle='--', color='b')
    ax_PCA[0].set_xlabel('Number of Principal Components')
    ax_PCA[0].set_ylabel('Cumulative variance (%)')
    ax_PCA[0].set_xticks(x)
    ax_PCA[0].set_title('Number of components\nneeded to explain variance')
    ax_PCA[0].axhline(y = threshold, color = 'r', linestyle = '--')
    ax_PCA[0].text(0.5, threshold + 0.01, f'{threshold*100}% total variance', color = 'red', fontsize = 10)
    ax_PCA[0].grid(axis = 'x')
    ax_PCA[1].bar(x, ExpVar)
    ax_PCA[1].set_title("Explained variance")
    ax_PCA[1].set_ylabel("Norm variance")
    ax_PCA[1].set_xlabel("#Component")
    [ax_PCA[2].scatter(df_eigen['PC1'][eigen], df_eigen['PC2'][eigen], color='k', alpha=0.4) for eigen in range(df_eigen.shape[0])]
    [ax_PCA[2].annotate(i, (df_eigen['PC1'][i], df_eigen['PC2'][i])) for i, txt in enumerate(range(df_eigen.shape[0]))]
    ax_PCA[2].set_title('PCA')
    ax_PCA[2].set_xlabel('PC1 ({:.2f}%)'.format(ExpVar[0]*100))
    ax_PCA[2].set_ylabel('PC2 ({:.2f}%)'.format(ExpVar[1]*100))
    ax_PCA[2].axvline(x=0.0,color='k',linestyle='--')
    ax_PCA[2].axhline(y=0.0,color='k',linestyle='--')
    plt.savefig(savedir+f'\{delay}_{protocol}\PCA_Summary_PYRAMIDAL_70.pdf')

    
    # plt.figure()
    # ax_pca3d = plt.axes(projection='3d')
    # [ax_pca3d.scatter3D(df_eigen['PC1'][eigen], df_eigen['PC2'][eigen], df_eigen['PC3'][eigen], color='k', alpha=0.4) for eigen in range(df_eigen.shape[0])]
    # [ax_pca3d.text(x, y, z, label) for x, y, z, label in zip(df_eigen['PC1'], df_eigen['PC2'], df_eigen['PC3'], range(df_eigen.shape[0]))]
    # ax_pca3d.set_title('PCA')
    # ax_pca3d.set_xlabel('PC1 ({:.2f}%)'.format(ExpVar[0]*100))
    # ax_pca3d.set_ylabel('PC2 ({:.2f}%)'.format(ExpVar[1]*100))
    # ax_pca3d.set_zlabel('PC3 ({:.2f}%)'.format(ExpVar[2]*100))
    fig_contribution, ax_contribution = plt.subplots(1,d,figsize=(17,4),sharey=True,tight_layout=True)
    for plot in range(d):
        ax_contribution[plot].set_title(f'PC{plot+1} contributions 70%Var')
        ax_contribution[plot].bar(x, VarComp[plot])
        ax_contribution[plot].set_xticks(x)
        ax_contribution[plot].set_xticklabels(df_pca.columns, rotation='vertical', fontsize=7)
        ax_contribution[plot].set_ylabel('Variable contribution')
        ax_contribution[plot].axhline(y=0.0, color='k', linestyle='--')
        
    plt.savefig(savedir+f'\{delay}_{protocol}\PCA_Summary_PC_Contribution_PYRAMIDAL_70.pdf')

    return ExpVar, df_eigen, pca



def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, delay=None, protocol=None, label_rotation=0, lims=None):
    """Display correlation circles, one for each factorial plane"""

    # For each factorial plane
    for d1, d2 in axis_ranks: 
        if d2 < n_comp:

            # Initialise the matplotlib figure
            fig, ax = plt.subplots(figsize=(10,10))

            # Determine the limits of the chart
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # Add arrows
            # If there are more than 30 arrows, we do not display the triangle at the end
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (see the doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # Display variable names
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # Display circle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # Define the limits of the chart
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # Display grid lines
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title(f"Correlation Circle (PC{d1+1} and PC{d2+1}) {delay} {protocol} ONLY PYRAMIDAL")
            plt.show(block=False)
            # plt.savefig(savedir+f'\Correlation_circle\CorrCircle_{delay}_{protocol}_PC{d1+1}_PC{d2+1}_PYRAMIDAL.pdf')
            
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def do_hcpc(df, principalDf, ExpVar, N_CLUST, protocol, delay):
    principalComponents = np.array(principalDf)
    #HAC-PCA based with n_components and n_clusters settings
    HCPC = AgglomerativeClustering(n_clusters=N_CLUST, linkage='ward',compute_distances=True)
    HCPC_clusters = HCPC.fit_predict(principalDf)
    clr = []
    cmap = cm.get_cmap('brg', N_CLUST)
    for c in range(cmap.N):
        rgba = cmap(c)
        clr.append(colors.rgb2hex(rgba))
    
    plt.figure(figsize=(12,10))
    for i in range(len(principalComponents)):
        plt.scatter(principalComponents[i][0], principalComponents[i][1], color=clr[HCPC_clusters[i]], alpha=0.5, label=f'Cluster {HCPC_clusters[i]}')
    
    plt.xlabel('PC1 ({:.2f}%)'.format(ExpVar[0]*100))
    plt.ylabel('PC2 ({:.2f}%)'.format(ExpVar[1]*100))
    plt.title('HCPC after PCA 70%Var')
    plt.axvline(x=0.0,color='k',linestyle='--')
    plt.axhline(y=0.0,color='k',linestyle='--')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(savedir+f'\{delay}_{protocol}\HCPC_{N_CLUST}_Clusters_PYRAMIDAL_70.pdf')

    hcpc_df = pd.DataFrame([HCPC_clusters.T,df.index], index=['HCPC_Cluster', 'Cluster_Name']).T

    plt.figure()
    model_for_dendro = HCPC.fit(principalDf)
    plt.title('Hierarchical Clustering Dendrogram 70%Var')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model_for_dendro, truncate_mode=None)#'level', p=N_CLUST)
    plt.xlabel("Points ID from PCA space")
    plt.savefig(savedir+f'\{delay}_{protocol}\HCPC_{N_CLUST}_Clusters_PYRAMIDAL_DENDROGRAM_70.pdf')
    return hcpc_df,HCPC_clusters


def plot_hcpc(df, hcpc_df,HCPC_clusters, ExpVar,delay,protocol,N_CLUST,savedir=None):
    # Define n components
    # expvar = ExpVar*100
    # cumulative = np.cumsum(expvar)
    # plt.figure(figsize=(12,10))
    # plt.plot(cumulative)
    # plt.ylabel('Variance Explained (%)')
    # plt.xlabel('PC')
    # plt.savefig(savedir+f'\{delay}_{protocol}\Explained_Variance_PYRAMIDAL.png')
    # plt.close()
    
    fig, ax = plt.subplots(2,2)
    fig.suptitle(f'{delay} {protocol} ONLY PYRAMIDAL 70%Var')
    
    for x in range(N_CLUST):
        clust_labels = hcpc_df.loc[hcpc_df['HCPC_Cluster']==x].Cluster_Name
        new_df = df[clust_labels].T
    
        # print(f'CLUSTER: {x}', '\n', clust_labels)
        mean = np.mean(new_df,axis=0)
        sem = stats.sem(new_df,axis=0)
        time = np.linspace(0,9,len(mean))
        smooth = gaussian_filter1d(mean,sigma=1.5)
        ax = ax.ravel()
        
        ax[x].plot(time, mean)
        ax[x].fill_between(time, mean+sem, mean-sem, alpha=0.2)
        ax[x].plot(time, smooth)
        ax[x].axvspan(0, 0.5,color='g', alpha = 0.2) # Cue1
        ax[x].axvspan(1.5, 2,color='g', alpha = 0.2) #Cue2
        ax[x].axvline(1.03, color='r', linestyle='--')
        if delay == 'fixed':
            ax[x].axvspan(2.53, 2.68, color='r', alpha=0.2)
        else:
            ax[x].axvspan(2.43,2.93, color='r', alpha=0.2)            
        ax[x].set_title(f'Group {x}')
    plt.savefig(savedir+f'\{delay}_{protocol}\Mean_FR_{N_CLUST}_Clusters_PYRAMIDAL_70.pdf')


###############################################################################
#####################      CLUSTERING UNITS    ############################
###############################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from matplotlib import cm
from matplotlib import colors
from collections import OrderedDict
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from matplotlib.collections import LineCollection
import extrapy.Organize as og
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler 


if __name__ == '__main__':
    
    protocol = 'all_stim'
    delay = 'random'
    
    savedir = r'D:\F.LARENO.FACCINI\RESULTS\New Results\Spike Sorting\Spike Times\Sorted Spikes\Clustering\Figures'
    path = fr'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/Clustering/Datasets for PCA/PCA/{delay}_{protocol}_values_pca.csv'
    df = pd.read_csv(path,index_col='Unnamed: 0')

    # =============================================================================
    #  REMOVE INTERNEURONS
    int_labels = [i for i in df.index if 'Interneuron' in i]
    print(len(int_labels))
    df.drop(labels=int_labels, inplace=True)
    # =============================================================================

    fr_path = fr'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/Clustering/Datasets for PCA/{delay}_{protocol}_fr.csv'
    fr_df = pd.read_csv(fr_path,index_col='Unnamed: 0').T

    N_CLUST = 4
    
    ExpVar, principalDf, pca = PrinCompAn(df,0.7)
    hcpc_df,HCPC_clusters = do_hcpc(df, principalDf, ExpVar, N_CLUST, protocol, delay)
    og.pickle_saving(fr'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Sorted Spikes/Clustering/Datasets for PCA/Definitive_Labels_HCPC/Labels_{delay}_{protocol}_{N_CLUST}_clusters_PYRAMIDAL_70', hcpc_df)

    plot_hcpc(fr_df, hcpc_df,HCPC_clusters, ExpVar,delay,protocol,N_CLUST, savedir)
    
    
    # # Generate a correlation circle
    # pcs = pca.components_ 
    # display_circles(pcs, n_components, pca, [(0,1)], labels = np.array(df.columns),delay=delay,protocol=protocol)

    
    
    
    
    
    
