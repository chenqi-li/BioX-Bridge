# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import joblib
import os
import sys
sys.path.append("../../papagei-foundation-model/")
from utilities import get_data_info
from tqdm import tqdm 

def compute_signal_statistics_df(df, path, case_name):
    case_ids = np.unique(df.loc[:, case_name].values)  
    signal_feats = {}
    print(case_ids)
    for i in tqdm(range(len(case_ids))):
        case = str(case_ids[i])
        segments = df[df[case_name] == case_ids[i]].segments.values
        subdir = os.path.join(path, case)
        signal = []
    
        for seg in segments: 
            segment = joblib.load(os.path.join(subdir, seg))
            signal.append(segment)
            
        signal = np.vstack(signal)
        mean = np.mean(signal, axis=1)
        std = np.std(signal, axis=1)
        percentile_25 = np.percentile(signal, 25, axis=1)
        percentile_50 = np.percentile(signal, 50, axis=1)
        percentile_75 = np.percentile(signal, 75, axis=1)
        minimum = np.min(signal, axis=1)
        maximum = np.max(signal, axis=1)

        signal_feats[case_ids[i]] = np.hstack([np.mean(mean), 
                                               np.mean(std), 
                                               np.mean(percentile_25), 
                                               np.mean(percentile_50), 
                                               np.mean(percentile_75), 
                                               np.mean(minimum), 
                                               np.mean(maximum)])
    return signal_feats

def compute_signal_statistics_df_segment(df, path, case_name):
    case_ids = np.unique(df.loc[:, case_name].values)  
    signal_feats = {}
    for i in tqdm(range(len(case_ids))):
        case = str(case_ids[i])
        segments = df[df[case_name] == case_ids[i]].segments.values
        subdir = os.path.join(path, case)
        signal = []
    
        for seg in segments: 
            segment = joblib.load(os.path.join(subdir, str(seg)))
            signal.append(segment)

        signal = np.vstack(signal)
        print(signal)
        mean = np.mean(signal, axis=1)
        std = np.std(signal, axis=1)
        percentile_25 = np.percentile(signal, 25, axis=1)
        percentile_50 = np.percentile(signal, 50, axis=1)
        percentile_75 = np.percentile(signal, 75, axis=1)
        minimum = np.min(signal, axis=1)
        maximum = np.max(signal, axis=1)

        signal_feats[case_ids[i]] = np.vstack([mean, 
                                               std, 
                                               percentile_25, 
                                               percentile_50, 
                                               percentile_75, 
                                               minimum, 
                                               maximum]).T
    return signal_feats