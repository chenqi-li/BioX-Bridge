# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np
import os
import joblib
from tqdm.auto import tqdm

def waveform_to_segments(waveform_name, segment_length, clean_signal=None, dict_data=None):
    """
    Select the waveform from the dict_data and segment the waveform based on length

    Args:
        dict_data (dictionary): The dictionary contains multiple waveforms under different keys.
        waveform_name (string): waveform to segment. For example, "ppg"
        segment_length (int) : length of segment calculated as frequency (Hz) * time (s)
    
    Returns:
        dict_data (dictionary): Dictionary with the segmented waveform
    """
    if dict_data is not None:
        signal = dict_data[waveform_name]
        if signal.ndim == 1:
            segments = np.array([signal[i:i+segment_length] for i in range(0, len(signal), segment_length)][:-1])
            dict_data[waveform_name] = segments
        else:
            print("[INFO] Already segmented -- skipping")
        return dict_data
    
    else:
        signal = clean_signal
        if signal.ndim == 1:
            segments = np.array([signal[i:i+segment_length] for i in range(0, len(signal), segment_length)][:-1])
        else:
            print("[INFO] Already segmented -- skipping")
        return segments


def segment_waveforms(path, waveform_name, segment_length):
    """
    Re-factor... 
    """
    
    filenames = os.listdir(path)
    for i in tqdm(range(len(filenames))):
        f_name = filenames[i]
        print(f"Segmenting file: {f_name} | {i}/{len(filenames)}")
        try:
            dict_data = joblib.load(os.path.join(path, f_name))
            dict_data = waveform_to_segments(dict_data=dict_data,
                                            waveform_name=waveform_name,
                                            segment_length=segment_length)
            joblib.dump(dict_data, os.path.join(path, f_name))
        except EOFError as e:
            print(f"[ERROR] {f_name}")

def save_segments_to_directory(save_dir, dir_name, segments, content=""):
    """
    Save segmented ppg to a directory
    
    Args:
        save_dir (string): Parent directory to save the segments
        dir_name (string): child directory, usually create one for each participant
        segments (np.array): 2D array ppg segment array (no. of segments x length of segment)
    """
    
    if not os.path.exists(os.path.join(save_dir, dir_name)):
        os.mkdir(os.path.join(save_dir, dir_name))
        print(f"[INFO] Saving segments to {dir_name}")
        for i in range(len(segments)):
            joblib.dump(segments[i], os.path.join(save_dir, dir_name, content + str(i) + ".p"))
    else:
        print(f"[INFO] {dir_name} already exists")

if __name__ == "__main__":

    path = "../data/vitaldb/"
    waveform_name = "ppg"
    frequency = 500
    segment_time = 10
    segment_length = frequency * segment_time

    segment_waveforms(path=path,
                    waveform_name=waveform_name,
                    segment_length=segment_length)