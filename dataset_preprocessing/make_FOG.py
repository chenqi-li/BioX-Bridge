import os
import random
import pickle
import shutil

import scipy.io


from multiprocessing import Pool
import numpy as np
import mne
import sys
import matplotlib.pyplot as plt

import argparse

from scipy.signal import resample
from biosppy.signals.tools import filter_signal
from scipy.interpolate import interp1d
from tqdm import tqdm
import pandas as pd
from collections import Counter
from scipy import signal
import torch

random.seed(6)

# FOG Preprocessing
# Split each session into 3s segments with sliding step size of 0.3 seconds.

# Running this file: 
# Change the settings in args
# (BioX-Bridge) [anonymous@server /users/anonymous/BioX-Bridge/dataset_preprocessing]$ python make_FOG.py


def wt(ts, lf=0.1, hf=65, wl='gaus1', method='fft'):
    # in: L
    # out: FxL
    # cwtmatr, freqs = pywt.cwt(ts, np.arange(lf, hf), wl, method=method)
    cwtmatr = signal.cwt(ts, signal.ricker, np.arange(lf, hf))
    return cwtmatr #[F, L]

def spec_cwt(audio_data): # [nvar, L]
    x1 = audio_data[:, 1:] - audio_data[:, :-1]
    x2 = x1[:, 1:] - x1[:, :-1]

    all_specs = list()
    for c_i in range(audio_data.shape[0]):
        all_specs.append(np.stack([
            wt(audio_data[c_i, 2:]).T, # [L, n_mels]
            wt(x1[c_i, 1:]).T, 
            wt(x2[c_i]).T
        ])) # [3, L, n_mels]

    all_specs = np.stack(all_specs) # [nvar, 3, L, n_mels]

    return all_specs


# Function from ECG FM for preprocessing
def resample_ecgfm(feats, curr_sample_rate, desired_sample_rate):
    """
    Resample an ECG using linear interpolation.
    """
    if curr_sample_rate == desired_sample_rate:
        return feats

    desired_sample_size = int(
        feats.shape[-1] * (desired_sample_rate / curr_sample_rate)
    )

    x = np.linspace(0, desired_sample_size - 1, feats.shape[-1])

    return interp1d(x, feats, kind='linear')(np.arange(desired_sample_size))

def lead_std_divide(feats, constant_lead_strategy='zero'):
    # Calculate standard deviation along axis 1, keep dimensions for broadcasting
    std = feats.std(axis=-1, keepdims=True)
    std_zero = std == 0

    # Check if there are any zero stds or if strategy is 'nan'
    if not std_zero.any() or constant_lead_strategy == 'nan':
        # Directly divide, which will turn constant leads into NaN if any
        feats = feats / std

        return feats, std

    # Replace zero standard deviations with 1 temporarily to avoid division by zero
    std_replaced = np.where(std_zero, 1, std)
    feats = feats / std_replaced

    if constant_lead_strategy == 'zero':
        # Replace constant leads to be 0
        zero_mask = np.broadcast_to(std_zero, feats.shape)
        feats[zero_mask] = 0

    elif constant_lead_strategy == 'constant':
        # Leave constant leads as is
        pass

    else:
        raise ValueError("Unexpected constant lead strategy.")

    return feats, std

def split_into_segments(data, labels, sf, segment_length, step_size):
    """Split data into segments of specified length (in seconds), with adjacent segments shifted by step_size seconds.
    Keep only segments where one label forms at least 80% of the segment.
    """
    segment_samples = int(segment_length * sf)
    step_samples = int(step_size * sf)
    num_segments = (data.shape[-1] - segment_samples) // step_samples + 1
    segments = []
    segment_labels = []

    for i in range(num_segments):
        start = i * step_samples
        end = start + segment_samples
        segment = data[:, start:end]
        segment_label = labels[start:end]

        # Count labels and their frequencies using NumPy
        unique_labels, counts = np.unique(segment_label, return_counts=True)
        majority_index = np.argmax(counts)
        majority_label = unique_labels[majority_index]
        majority_ratio = counts[majority_index] / len(segment_label)

        # Keep the segment only if the majority label occurs in at least 80% of the segment and is 1, 2, or 3
        if majority_ratio >= 0.8:
            segments.append(segment)
            segment_labels.append(majority_label)

    return np.array(segments), np.array(segment_labels)

# Preprocess, Save
def preprocess_split_save(args, file):
    # Load basic variables
    desired_emg_rate = 130
    desired_eeg_rate = 200

    # Load data
    df = pd.read_csv(file, header=None)
    eeg_data = df.iloc[:, 2:27].to_numpy().T # column 2-26 is EEG
    eeg_data = eeg_data[[2, 3, 4, 5, 8, 9],:] # keep only F3, C3, O1, F4, C4, O2
    column_map = {
        '001': [27, 28, 31],
        '002': [27, 28, 31],
        '006': [27, 28, 31],
        '007': [27, 28, 31],
        '008/OFF_1': [27, 28, 31],
        '003': [28, 27, 31],
        '004': [28, 27, 31],
        '005': [28, 27, 31],
        '010': [28, 27, 31],
        '011': [28, 27, 31],
        '012': [28, 27, 31],
        '008/OFF_2': [27, 31, 28],
        '009': [28, 27, 30],
    }
    parts = file.split(os.sep)
    folder_id = os.path.join(parts[-3], parts[-2]) if parts[-2].startswith("OFF") else parts[-2]
    cols = column_map[folder_id]
    emg_data = df.iloc[:, cols].to_numpy().T
    labels = df.iloc[:, 60].to_numpy().T

    # Preprocess EEG
    if args.eegfm in ['labram']:
        # # Optional plotting
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        # ax1.plot(eeg_data[0, :1000], color='b')
        # ax1.set_title('Original EEG Data')
        # ax1.set_xlabel('Time (samples)')
        # ax1.set_ylabel('Amplitude')
        
        eeg_data = resample_ecgfm(eeg_data, curr_sample_rate=500, desired_sample_rate=desired_eeg_rate) # No need to convert unit to 0.1mV for range [-1, 1] since that is done in train_one_epoch of LaBraM
        if args.eegznorm == 1:
            mean = eeg_data.mean(axis=-1, keepdims=True)
            eeg_data = eeg_data - mean
            eeg_data, std = lead_std_divide(
                eeg_data,
                constant_lead_strategy='zero',
            )
        # # Optional plotting
        # ax2.plot(eeg_data[0, :400], color='r')
        # ax2.set_title('Processed EEG Data')
        # ax2.set_xlabel('Time (samples)')
        # ax2.set_ylabel('Amplitude')
        # output_filename = 'eeg_plot.png'
        # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        # plt.close()
    else:
        print(f'{args.eegfm} preprocessing not implemented!')

    # Preprocess EMG
    # NormWear: We follow their preprocessing code exactly
    if args.emgfm in ['normwear']:
        # # Optional plotting
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        # ax1.plot(emg_data[0, :1000], color='b')
        # ax1.set_title('Original EMG Data')
        # ax1.set_xlabel('Time (samples)')
        # ax1.set_ylabel('Amplitude')

        emg_data = resample_ecgfm(emg_data, curr_sample_rate=500, desired_sample_rate=desired_emg_rate)
        mean = emg_data.mean(axis=-1, keepdims=True)
        emg_data = emg_data - mean
        emg_data, std = lead_std_divide(
            emg_data,
            constant_lead_strategy='zero',
        )

        # # Optional plotting
        # ax2.plot(emg_data[0, :260], color='r')
        # ax2.set_title('Processed EMG Data')
        # ax2.set_xlabel('Time (samples)')
        # ax2.set_ylabel('Amplitude')
        # output_filename = 'emg_plot.png'
        # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        # plt.close()
    else:
        print(f'{args.eegfm} preprocessing not implemented!')

    # Split into segments and save    
    eeg_segments, eeg_segment_labels = split_into_segments(eeg_data, np.round(signal.resample(labels, eeg_data.shape[-1])).astype(int), desired_eeg_rate, args.segment_length, args.step_size)
    emg_segments, emg_segment_labels = split_into_segments(emg_data, np.round(signal.resample(labels, emg_data.shape[-1])).astype(int), desired_emg_rate, args.segment_length, args.step_size)
    assert eeg_segments.shape[0] == emg_segments.shape[0], f"EEG and EMG segments do not match. EEG: {eeg_segments.shape[0]}, EMG: {emg_segments.shape[0]}"
    assert eeg_segment_labels.shape[0] == emg_segment_labels.shape[0]
    if not np.array_equal(eeg_segment_labels, emg_segment_labels):
        raise ValueError("EEG and EMG labels do not match. Check segmentation logic.")
    segment_labels = eeg_segment_labels

    # CWT for EMG
    processed_emg_segments = np.stack([spec_cwt(sample) for sample in emg_segments], axis=0)
    emg_segments = processed_emg_segments

    for i in range(eeg_segments.shape[0]):
        # Save the segment
        file_name = f"subject_{str(folder_id).replace('/','_')}_{parts[-1].split('.')[0]}_segment_{str(i).zfill(4)}"
        eeg_segment = eeg_segments[i, :] # [nvar, L]
        label_segment = segment_labels[i]
        emg_segment = emg_segments[i, :] # [nvar, 3, L, n_mel]
        # Print how many segments is from each class using np.unique
        np.savez(os.path.join(args.output_dir, 'all', file_name), EEG=eeg_segment, EMG=emg_segment, label=label_segment)
        
    unq, cnt = np.unique(segment_labels, return_counts=True)
    print(f"File: {file}, We have {cnt} segments from {unq} our mapped classes")


if __name__ == "__main__":
    """
    FOG dataset is downloaded from https://data.mendeley.com/datasets/r8gmbtv7w2/3
    """
    
    # Define configurations
    parser = argparse.ArgumentParser(description="Configuration for Processing")
    parser.add_argument("--fog_path", type=str, default="/data/anonymous/FOG", help="Path to the FOG dataset")
    parser.add_argument("--splits", type=dict, default={"train": 33, "val": 11, "pair": 33, "test": 22}, help="Ratio for each of the splits")
    parser.add_argument("--eegfm", type=str, default='labram', choices=['labram'])
    parser.add_argument("--emgfm", type=str, default='normwear', choices=['normwear'])
    parser.add_argument("--segment_length", type=int, default=3, help='Length of each segment in seconds')
    parser.add_argument("--step_size", type=float, default=0.3, help='Step size between adjacent segments in seconds')
    parser.add_argument("--eegznorm", type=int, default=0, help='Whether to do z-score norm for EEG: 0 - No norm, 1 - Norm')
    parser.add_argument("--num_classes", type=int, default=2, help='Do freeze of gait classification: 0 - No FOG, 1 - FOG')
    parser.add_argument("--subject_filter", type=str, default='008', help='Select one subject from 001-012, we use subject 008 for our experiments')
    args = parser.parse_args()

    # Define output directory based on configurations
    if args.eegznorm == 1:
        args.output_dir = os.path.join(args.fog_path, f'processed_eegfm{args.eegfm}_emgfm{args.emgfm}_eegznorm_subject{args.subject_filter}_{args.num_classes}classes')
    else:
        args.output_dir = os.path.join(args.fog_path, f'processed_eegfm{args.eegfm}_emgfm{args.emgfm}_subject{args.subject_filter}_{args.num_classes}classes')

    # Setup the output folders
    folders = args.splits.keys()
    for folder in folders: # folders for the train, val, pair, test splits
        if not os.path.exists(os.path.join(args.output_dir, folder)):
            os.makedirs(os.path.join(args.output_dir, folder))
    if not os.path.exists(os.path.join(args.output_dir, 'all')): # folder for all segments, before they are split into the splits
        os.makedirs(os.path.join(args.output_dir, 'all'))

    # Initialize the dataset_summary.txt and redirect all prints to the text file, will be restored to stdout later
    output_file_path = os.path.join(args.output_dir, "dataset_summary.txt")
    sys.stdout = open(output_file_path, 'w', buffering=1)

    # List all the files
    txt_files = []
    for root, dirs, files in os.walk(os.path.join(args.fog_path, 'FilteredData')):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    
    # Filter the subject
    txt_files = [f for f in txt_files if args.subject_filter in f]
    print(txt_files)

    # Preprocess, Save
    for file in tqdm(txt_files):
        preprocess_split_save(args, file)

    # Split the segments into different splits
    all_dir = os.path.join(args.output_dir, "all")
    split_dirs = {k: os.path.join(args.output_dir, k) for k in args.splits}
    all_files = [
        f for f in os.listdir(all_dir)
        if os.path.isfile(os.path.join(all_dir, f)) and f.endswith(".npz")
    ]
    random.shuffle(all_files)
    total_split_units = sum(args.splits.values())
    total_files = len(all_files)
    split_sizes = {
        k: int(round((v / total_split_units) * total_files))
        for k, v in args.splits.items()
    }
    start = 0
    for split, size in split_sizes.items():
        end = start + size
        for f in all_files[start:end]:
            src = os.path.join(all_dir, f)
            dst = os.path.join(split_dirs[split], f)
            shutil.copy2(src, dst)
        start = end
    print(f'Split sizes: {split_sizes}')
    # Print arguments used
    print(args)

    # Restore original stdout at the end of the script
    sys.stdout.close()
    sys.stdout = sys.__stdout__
