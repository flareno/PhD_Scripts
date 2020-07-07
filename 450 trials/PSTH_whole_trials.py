# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:56:24 2019

@author: F.LARENO-FACCINI
"""

import matplotlib.pyplot as plt
import extrapy.Behaviour as bv
import glob
import os

files = glob.glob(r"D:/F.LARENO.FACCINI/Preliminary Results/Behaviour/Group 14/6401/Fixed Delay/P13/*.lick")
names = [os.path.basename(x) for x in files]

for x in range(len(names)):
    names[x] = names[x].replace(".lick","")


for file in (names):
    path = f"D:/F.LARENO.FACCINI/Preliminary Results/Behaviour/Group 14/6401/Fixed Delay/P13/{file}.lick"

    fig, ax = plt.subplots(2, sharex=True)
    
    lick = bv.load_lickfile(path)
    
    bv.scatter_lick(lick, ax=ax[0])
    bv.PSTH_lick(lick, color='r', ax=ax[1], lentrial=10)
    d=0.5
    
    ax[0].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
    ax[0].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
    ax[0].axvspan(2.07+d, 2.22+d,color='r', alpha = 0.2) # Reward delivery
    ax[0].axvspan(2.22+d,2.72+d,color='purple', alpha=0.2) # Aspiration
    ax[1].axvspan(0, 0.5,color='g', alpha = 0.2) # Light 1
    ax[1].axvspan(1.5, 2,color='g', alpha = 0.2) # Light 2
    ax[1].axvspan(2.07+d, 2.22+d,color='r', alpha = 0.2) # Reward delivery
    ax[1].axvspan(2.22+d,2.72+d,color='purple', alpha=0.2) # Aspiration
    ax[1].axvline(2.07+d, linestyle='--', color='k', linewidth=0.7, alpha=0.4)
    ax[1].set_ylabel('Number count')
    
    # fig.savefig(r"C:\Users\F.LARENO-FACCINI\Desktop\Blitz plots\blitz_{}.pdf".format(file))
    # plt.close()
