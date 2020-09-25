#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 09:27:25 2019

@author: Federica LARENO FACCINI

Create excel file as DATABASE of BEHAVIOURAL DATA.
One file for each animal.
The first time it runs for one animal it creates the file (first session). After that, everytime it runs, it adds a new sheet for the new behaviour session.
The ENVELOPE (n by bin of the PSTH by session)is saved in a specific sheet, one column for each session.
"""
import extrapy.Behaviour as bv
import pandas as pd
import numpy as np
import os
from openpyxl import load_workbook


mouse = 1094
session = 3

path = r"D:/F.LARENO.FACCINI/Preliminary Results/Scripts/New Pipeline/1094_2019_12_12_15_09_05/1094_2019_12_12_11_02_20.lick"
param = r"D:/F.LARENO.FACCINI/Preliminary Results/Scripts/New Pipeline/1094_2019_12_12_15_09_05/1094_2019_12_12_11_02_20.param"

# ========================================================================================
# Load lick files and random delays
# ========================================================================================
licks = bv.load_lickfile(path)
random = bv.extract_random_delay(param)

# ========================================================================================
# Format the lists
# ========================================================================================
delay = np.asarray([d for d, _ in (random)])
trial = np.asarray([d for _,d in (random)])

# ========================================================================================
# Divide licks by trial and get number of licks per trial
# ========================================================================================
licks_by_trial = []
num_licks = []

N_TRIALS = int(max(licks[:,0]))

for i in range(N_TRIALS):
    temp__ = [licks[:,1][j] for j in range(len(licks)) if licks[:,0][j]==i+1]
    num_licks.append(len(temp__))
    if len(temp__) > 0:
        licks_by_trial.append(temp__)
    else:
        licks_by_trial.append(np.nan)

# ========================================================================================
# Extracting the envelope of the PSTH (the n of the PSTH)
# ========================================================================================
n,bins,patches = bv.psth_lick(licks)
col =[f'Session {session}']
env_df = pd.DataFrame(n, index=None, columns=col)

# ========================================================================================
# Create the dataframe
# ========================================================================================
cols = ['Trial Number', 'Delay (ms)', 'Number of Licks', 'Licks']
df = pd.DataFrame(zip(trial, delay, num_licks, licks_by_trial),columns=cols)


# ========================================================================================
# Create Excel File
# ========================================================================================
savedir = r"D:\F.LARENO.FACCINI\Preliminary Results\Scripts\New Pipeline"

#if the file already exists, append a new sheet. To have one file per animal with the different sessions in different sheets
if os.path.isfile(f'{savedir}\{mouse}.xlsx'):
    book = load_workbook(f'{savedir}\{mouse}.xlsx')
    exc = pd.read_excel(f'{savedir}\{mouse}.xlsx', sheet_name='Envelope')
    exc[f'Session {session}'] = n
    
    with pd.ExcelWriter(f'{savedir}\{mouse}.xlsx', engine="openpyxl") as writer:
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        exc.to_excel(writer, index=False, sheet_name='Envelope')
        df.to_excel(writer, index=False, sheet_name=f"Session {session}")

        
        
# creates new excel file if it doesn't exist for that mouse
else:
    with pd.ExcelWriter(f'{savedir}\{mouse}.xlsx', engine="openpyxl") as writer:    
        env_df.to_excel(writer,index=False, columns = col, sheet_name='Envelope')
        df.to_excel(writer, index=False, sheet_name=f"Session {session}")
