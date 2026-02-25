# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pandas as pd
import numpy as np
import joblib 
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
from scipy.stats import kurtosis, skew, entropy
from torch_ecg._preprocessors import Normalize
from scipy.signal import argrelmax, argrelmin
from scipy.integrate import trapz

def extract_svri(single_waveform):
    """
    https://github.com/qiriro/PPG

    Args:
        single_waveform (np.array): input ppg segment
    Returns:
        svri (float): svri value
    """
    def __scale(data):
        data_max = max(data)
        data_min = min(data)
        return [(x - data_min) / (data_max - data_min) for x in data]
    max_index = np.argmax(single_waveform)
    single_waveform_scaled = __scale(single_waveform)
    return np.mean(single_waveform_scaled[max_index:]) / np.mean(single_waveform_scaled[:max_index])

def skewness_sqi(x, axis=0, bias=True, nan_policy='propagate'):
    """
    Computes ppg skewness using skew from scipy
    """
    return skew(x, axis, bias, nan_policy)

def compute_ipa(signal, fs):
    """
    Computes IPA by identifying the first dicrotic notch

    Args:
        signal(np.array): input ppg segment
        fs (int): ppg frequency
    Returns:
        ipa (float): IPA value
    """

    try:
        maxima_index = argrelmax(signal, order=fs // 5)[0]
        minima_index = argrelmin(signal, order=fs // 5)[0]
        
        single_beat = signal[minima_index[0]:minima_index[1]]
        minima_beats = argrelmin(single_beat)[0]
    
        minima_beat = minima_beats[0]
        
        sys_values = single_beat[:minima_beat]
        dias_values = single_beat[minima_beat:]
        
        sys_x_values = np.linspace(0, len(sys_values) - 1, len(sys_values)) 
        dias_x_values = np.linspace(0, len(dias_values) - 1, len(dias_values)) 
        
        sys_phase = trapz(y=sys_values, x=sys_x_values)
        dias_phase = trapz(y=dias_values, x=dias_x_values)
        ipa = sys_phase/dias_phase
        
    except IndexError as e:
        ipa = 0 
        
    return ipa


def compute_features_for_dataset(main_dir, save_dir, fs, columns):
    """
    Extract sVRI and SQI from PPG in batches

    Args:
        main_dir (string): Location of ppg segments
        save_dir (string): directory to save dataframe after computing morphology
        fs (int): ppg frequency
        columns (list): columns to create the dataframe
        
    """
    patients_dir = os.listdir(main_dir)
    patient_seg = {}
    pattern = r'_(.+)'
    
    for i in tqdm(range(len(patients_dir))):
        patient = patients_dir[i]
        segments = os.listdir(os.path.join(main_dir, patient))
        
        if i % 100 == 0 and i != 0:
            print(f"Saving morphology {i}")
            patients_df = [s.split("_")[0] for s in list(patient_seg.keys())]
            segments_df = [re.search(pattern, s).group(1) for s in list(patient_seg.keys())]
            df = pd.DataFrame(data=patient_seg.values(), columns=columns)
            df.insert(0, column='case_id', value=patients_df)
            df.insert(1, column='segment', value=segments_df)
            df.to_csv(f"{save_dir}/morphology/morphology_{str(i)}.csv", index=False)
            patient_seg = {}
            
        for s in segments:
            ppg = joblib.load(os.path.join(main_dir, patient, s))
            svri = extract_svri(ppg)
            ppg = np.vstack([ppg[p:p+5*fs] for p in range(0, len(ppg), 5*fs)])
            sqi = np.mean(skewness_sqi(ppg, axis=1))

            patient_seg[f"{patient}_{s}"] = [svri, sqi]
    
    # Save any remaining patient data that hasn't been saved yet
    if patient_seg:
        print(f"Saving final morphology batch")
        patients_df = [s.split("_")[0] for s in list(patient_seg.keys())]
        segments_df = [re.search(pattern, s).group(1) for s in list(patient_seg.keys())]
        df = pd.DataFrame(data=patient_seg.values(), columns=columns)
        df.insert(0, column='case_id', value=patients_df)
        df.insert(1, column='segment', value=segments_df)
        df.to_csv(f"{save_dir}/morphology/morphology_final.csv", index=False)
    
    return patient_seg

def compute_ipa_for_dataset(main_dir, save_dir, fs, columns):
    """
    Extract IPA from PPG
    """
    patients_dir = os.listdir(main_dir)
    patient_seg = {}
    pattern = r'_(.+)'
    norm = Normalize(method='z-score')

    for i in tqdm(range(len(patients_dir))):
        patient = patients_dir[i]
        segments = os.listdir(os.path.join(main_dir, patient))
    
        if i % 100 == 0 and i != 0:
            print(f"Saving morphology {i}")
            patients_df = [s.split("_")[0] for s in list(patient_seg.keys())]
            segments_df = [re.search(pattern, s).group(1) for s in list(patient_seg.keys())]
            df = pd.DataFrame(data=patient_seg.values(), columns=columns)
            df.insert(0, column='case_id', value=patients_df)
            df.insert(1, column='segments', value=segments_df)
            df.to_csv(f"{save_dir}/ipa/ipa_{str(i)}.csv", index=False)
            patient_seg = {}
            
        for s in segments:
            ppg = joblib.load(os.path.join(main_dir, patient, s))
            ppg, _ = norm.apply(ppg, fs)
            ipa = compute_ipa(ppg, fs)
            patient_seg[f"{patient}_{s}"] = ipa
    
    # Save any remaining patient data that hasn't been saved yet
    if patient_seg:
        print(f"Saving final morphology batch")
        patients_df = [s.split("_")[0] for s in list(patient_seg.keys())]
        segments_df = [re.search(pattern, s).group(1) for s in list(patient_seg.keys())]
        df = pd.DataFrame(data=patient_seg.values(), columns=columns)
        df.insert(0, column='case_id', value=patients_df)
        df.insert(1, column='segments', value=segments_df)
        df.to_csv(f"{save_dir}/ipa/ipa_final.csv", index=False)
    
    return patient_seg
