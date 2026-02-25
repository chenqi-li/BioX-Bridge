# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import biobss
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
import joblib
import argparse
from joblib import Parallel, delayed
from scipy import integrate
from tqdm import tqdm 
from torch_ecg._preprocessors import Normalize

def is_signal_flat_lined(sig, fs, flat_time, signal_time, flat_threshold=0.25, change_threshold=0.01):

    signal_length = fs * signal_time
    flat_segment_length = fs * flat_time
    norm = Normalize(method='z-score')
    norm_sig = norm(sig, fs=fs)[0]

    flatline_segments = biobss.sqatools.detect_flatline_segments(sig, 
                                                                 change_threshold=change_threshold, 
                                                                 min_duration=flat_segment_length)
    
    total_flatline_in_signal = np.sum([end - start for start, end in flatline_segments])

    if total_flatline_in_signal / signal_length > flat_threshold:
        return 1
    else:
        return 0

def process_segment(p, s, main_dir, fs, flat_time, signal_time):
    sig = joblib.load(os.path.join(main_dir, p, s))
    flatline = is_signal_flat_lined(sig, fs, flat_time, signal_time)
    return p, s, flatline

def flat_line_check(main_dir, fs, flat_time, signal_time, start_idx, end_idx, n_jobs=-1):
    patients = os.listdir(main_dir)[start_idx:end_idx]
    
    results = Parallel(n_jobs=n_jobs)(delayed(process_segment)(p, s, main_dir, fs, flat_time, signal_time)
                                      for p in tqdm(patients)
                                      for s in os.listdir(os.path.join(main_dir, p)))
    
    df = pd.DataFrame(results, columns=['patients', 'segments', 'flatlined'])
    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("main_dir", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("fs", type=int)
    parser.add_argument("flat_time", type=int)
    parser.add_argument("signal_time", type=int)
    args = parser.parse_args()

    # Loop over start_idx and end_idx
    batch_size = 200
    total_files = 2200

    for start_idx in range(0, total_files, batch_size):
        end_idx = min(start_idx + batch_size, total_files)

        df = flat_line_check(main_dir=args.main_dir,
                        fs=args.fs,
                        flat_time=args.flat_time,
                        signal_time=args.signal_time,
                        start_idx=start_idx,
                        end_idx=end_idx)

        df.to_csv(f"{args.save_dir}/faltline_{start_idx}_{end_idx}.csv", index=False)

