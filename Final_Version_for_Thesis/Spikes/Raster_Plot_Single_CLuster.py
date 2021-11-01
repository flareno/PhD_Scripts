# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 08:51:32 2021

@author: F.LARENO-FACCINI
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import quantities as pq
from viziphant.rasterplot import rasterplot

path = r'D:/F.LARENO.FACCINI/RESULTS/New Results/Spike Sorting/Spike Times/Fixed Delay/6409-EF-P13-NoStim_Spike_times_changrp0.xlsx'
df = pd.read_excel(path,sheet_name='Cluster 0')

spikes = np.array(df).T *pq.s

rasterplot(spikes, s=4, c='black')

plt.axvspan(0,0.5, color='gray', alpha=0.2)
plt.axvspan(1.5,2, color='gray', alpha=0.2)
plt.axvspan(2.55,2.7, color='red', alpha=0.2)

plt.show()


