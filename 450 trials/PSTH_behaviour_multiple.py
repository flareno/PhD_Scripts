# -*- coding: utf-8 -*-
import numpy as np
from numpy import genfromtxt as gen
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import glob
import os
matplotlib.rcParams['pdf.fonttype'] = 42

"""
Created on Mon Apr 16 18:32:17 2018

@author: F.LARENO-FACCINI
"""

files = glob.glob(r"C:/Users/F.LARENO-FACCINI/Desktop/Blitz plots/exp/*.lick")
names = [os.path.basename(x) for x in files]


for x in range(len(names)):
    names[x] = names[x].replace(".lick", "")

################################# LICK SCATTER ################################

for file in range(len(names)):

    file_path = (
        r"C:\Users\F.LARENO-FACCINI\Desktop\Blitz plots\exp\%s.lick" % (names[file]))

    A = (pd.read_csv(file_path, sep="\t", header=None))
    A = np.array(A)
    B = [[A[i][0], float(A[i][1].replace(',', '.'))] for i in range(len(A))]
    B = np.array(B)

    Y = []
    ant = []

    N_TRIALS = 500
    step = 16
    bins = np.arange(0, N_TRIALS+step, step)
    binning = np.arange(0, 10.01, 0.01)                # 10 ms time resolution

    for i in range(len(bins)):

        if i == len(bins)-1:
            break

        else:
            start = bins[i]
            stop = bins[i+1]
            to_iterate = np.arange(start+1, stop+1, 1)
            print(i)

            fig, ax = plt.subplots(4, 4, figsize=(
                15, 6), facecolor='w', edgecolor='k', sharex=True, sharey=True)
            fig.add_subplot(111, frameon=False)
            ax = ax.ravel()

            for p in range(len(to_iterate)):

                temp__ = [B[:, 1][j]
                          for j in range(len(B)) if B[:, 0][j] == (to_iterate[p])]

                n, binns, patches = ax[p].hist(
                    temp__, bins=binning, rwidth=0.3)
                ax[p].vlines(x=(2.68, 2.68), ymin=0,
                             ymax=3, color="red", alpha=0.2)
                ax[p].vlines(x=(2.88, 2.88), ymin=0, ymax=3,
                             color="purple", alpha=0.2)
                ax[p].title.set_text('Trial %s' % (to_iterate[p]))

                if len(temp__) > 0:
                    Y.append(np.array(temp__))
                else:
                    Y.append(np.nan)

                ant.append(np.array(n))

            plt.tick_params(labelcolor='none', top='off',
                            bottom='off', left='off', right='off')
            plt.grid(False)
            plt.ylabel('Number Count')
            plt.xlabel('Time (s)')
            plt.tight_layout()
#                plt.savefig(r"D:\F.LARENO.FACCINI\Preliminary Results\DATA\Group 9\2535\Trial whole\%s_%s-%s.pdf" %(names[file],start,stop))
#                plt.close()

    plt.figure()
    tot = np.sum(ant, axis=0)
    plt.xlabel('Time (s)')
    plt.ylabel('Number Count')
    plt.bar(np.arange(0, 10, 0.01), tot, width=0.009)
    plt.vlines(x=(2.68, 2.68), ymin=0, ymax=(np.amax(tot)),
               color="red", alpha=0.8, linewidth=0.4)
    plt.vlines(x=(2.88, 2.88), ymin=0, ymax=(np.amax(tot)),
               color="purple", alpha=0.8, linewidth=0.4)
#    plt.savefig(r"D:\F.LARENO.FACCINI\Preliminary Results\DATA\Group 9\2535\Trial whole\%s_PSTH_tot.pdf" %(names[file]))
#    plt.close()

    plt.figure()
    plt.imshow(ant, cmap='jet', vmin=0, vmax=(np.amax(ant)/2))
    plt.colorbar()
    plt.xlabel('Time (cs)')
    plt.ylabel('Trial Number')
    plt.vlines(x=(268, 268), ymin=0, ymax=N_TRIALS,
               color="red", alpha=0.8, linewidth=0.4)
    plt.vlines(x=(288, 288), ymin=0, ymax=N_TRIALS,
               color="purple", alpha=0.8, linewidth=0.4)
#    plt.savefig(r"D:\F.LARENO.FACCINI\Preliminary Results\DATA\Group 9\2535\Trial whole\%s_density_flow.pdf" %(names[file]))
#    plt.close()

    plt.show()
