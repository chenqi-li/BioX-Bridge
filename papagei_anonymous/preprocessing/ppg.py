# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pandas as pd 
import torch 
import os 
import numpy as np
import pickle
import pyPPG.preproc as PP

from dotmap import DotMap
from vitaldb import VitalFile

def load_vitaldb_waveforms(path, name, frequency):
    """
    Loading waveforms from vitaldb containing .vital files

    Args:
        path (string): path to .vital files
        name (string): file name
        frequency (int): loading frequency of the waveforms
    
    Returns:
        dict_waveforms (dictionary): Dictionary containing upto 3 wavesforms ECG_II, ECG_V5, and PLETH
    """
    
    waveforms = np.array(['SNUADC/ECG_II', 'SNUADC/ECG_V5', 'SNUADC/PLETH'])
    vf = VitalFile(os.path.join(path, name))

    df = vf.to_pandas(waveforms, interval=1/frequency)
    data_available = [waveforms[i] for i in range(len(waveforms)) if df.loc[:, waveforms[i]].isna().sum() != len(df)]
    df = df.loc[:, data_available].dropna()
    assert df.index.is_monotonic_increasing

    columns = df.columns
    dict_waveforms = {c.split('/')[1]: df.loc[:, c].values for c in columns}
    
    return dict_waveforms

def preprocess_one_ppg_signal(waveform,
                          frequency,
                          fL=0.5, 
                          fH=12, 
                          order=4, 
                          smoothing_windows={"ppg":50, "vpg":10, "apg":10, "jpg":10}):
    
    """
    Preprocessing a single PPG waveform using py PPG.
    https://pyppg.readthedocs.io/en/latest/Filters.html
    
    Args:
        waveform (numpy.array): PPG waveform for processing
        frequency (int): waveform frequency
        fL (float/int): high pass cut-off for chebyshev filter
        fH (float/int): low pass cut-off for chebyshev filter
        order (int): filter order
        smoothing_windows (dictionary): smoothing window sizes in milliseconds as dictionary
    
    Returns:
        ppg (numpy.array): filtered ppg signal
        ppg_d1 (numpy.array): first derivative of filtered ppg signal
        ppg_d2 (numpy.array): second derivative of filtered ppg signal
        ppg_d3 (numpy.array): third derivative of filtered ppg signal

    """

    prep = PP.Preprocess(fL=fL,
                    fH=fH,
                    order=order,
                    sm_wins=smoothing_windows)
    
    signal = DotMap()
    signal.v = waveform
    signal.fs = frequency
    signal.filtering = True

    ppg, ppg_d1, ppg_d2, ppg_d3 = prep.get_signals(signal)

    return ppg, ppg_d1, ppg_d2, ppg_d3

def save_pickle_ppg(dict_waveforms, f_name):
    """
    Saves the waveform dictionary as a pickle file 

    Args:
        dict_waveforms (dictionary): Dictionary containing upto 3 wavesforms ECG_II, ECG_V, and PLETH
        f_name (string): file name
    """

    # Sometime PPG is not available 
    try:        
        ppg, ppg_d1, ppg_d2, ppg_d3 = preprocess_one_ppg_signal(waveform=dict_waveforms['PLETH'],
                                                                frequency=frequency)

        dict_waveforms['ppg'] = ppg
        dict_waveforms['ppg_d1'] = ppg_d1
        dict_waveforms['ppg_d2'] = ppg_d2
        dict_waveforms['ppg_d3'] = ppg_d3
        pickle.dump(dict_waveforms, open(save_path + f_name.split(".")[0] + ".p", "wb"))

    except KeyError as e:
        print(f"PLETH not found for {f_name}")


def preprocess_ppg(vital_path, save_path, frequency, overwrite=True):
    """
    Preprocess and filter all ppg signals in vitalDB

    Args:
        vital_path (string): Path to vital db files
        save_path (string): directory to save the dictionary waveforms
        frequency (int): waveform frequency
        overwrite (boolean): True to overwrite existing .p files

    """
    
    # pickle loading does not work sometimes 
    vital_files = os.listdir(vital_path)

    for i in range(len(vital_files)):
        f_name = vital_files[i]
        print(f"[INFO] Processsing {f_name} | {i}/{len(vital_files)}")
        if not f_name.endswith(".vital"):
            continue
        if not overwrite:
            saved_files = os.listdir(save_path)
            check_f_name = f_name.split(".")[0] + ".p"
            if check_f_name in saved_files:
                print(f"File {f_name} already exists -- Skipping")
            else:
                dict_waveforms = load_vitaldb_waveforms(path=vital_path,
                                                       name=f_name,
                                                       frequency=frequency)
                
                save_pickle_ppg(dict_waveforms=dict_waveforms,
                                f_name=f_name)
        else:
                dict_waveforms = load_vitaldb_waveforms(path=vital_path,
                                                       name=f_name,
                                                       frequency=frequency)
                save_pickle_ppg(dict_waveforms=dict_waveforms,
                                f_name=f_name)
        
if __name__ == "__main__":

    vital_path = "/mnt/vitaldb/physionet.org/files/vitaldb/1.0.0/vital_files/"
    save_path = "../../data/vitaldb/"
    frequency = 500
    overwrite = False

    preprocess_ppg(vital_path=vital_path,
              save_path=save_path,
              frequency=frequency,
              overwrite=overwrite)