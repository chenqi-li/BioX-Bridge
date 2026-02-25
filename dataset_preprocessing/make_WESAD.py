import os
import random
import pickle

import scipy.io

import re

from multiprocessing import Pool
import numpy as np
import mne
import pandas as pd
import sys
import matplotlib.pyplot as plt

import argparse

from scipy.signal import resample
from biosppy.signals.tools import filter_signal
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy import signal

from scipy.stats import zscore


random.seed(6)

# WESAD Preprocessing
# For ECG and PPG:
# Split each session into 60s segments.

# Running this file: 
# Change the settings in args
# (BioX-Bridge) [anonymous@server /users/anonymous/BioX-Bridge/dataset_preprocessing]$ python make_WESAD.py

# Constants
SF_ECG = 700  # Sampling frequency of chest sensor (ECG)
SF_BVP = 64     # Sampling frequency of wrist sensor (PPG/BVP)


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
    """Split data into segments of specified length (in seconds), with adjacent segments shifted by step_size seconds."""
    segment_samples = int(segment_length * sf)
    step_samples = int(step_size * sf)
    num_segments = (len(data) - segment_samples) // step_samples + 1
    segments = []
    segment_labels = []

    for i in range(num_segments):
        start = i * step_samples
        end = start + segment_samples
        segment = data[start:end]
        segment_label = labels[start:end]

        # Ensure the segment contains only one label and exclude the irrelevant ones
        if len(np.unique(segment_label)) == 1 and segment_label[0] in [1, 2, 3]:
            segments.append(segment)
            segment_labels.append(segment_label[0])  # Use the first label in the segment as an integer

    return np.array(segments), np.array(segment_labels)

# Preprocess, Split, Save
def preprocess_split_save(args, split, file):
    # Load data and calculate basic variables
    subject_id = int(file.replace("S", "").replace(".pkl", ""))
    data = pd.read_pickle(os.path.join(args.wesad_path, file, f'{file}.pkl'))
    ecg = data["signal"]["chest"]["ECG"].squeeze()  # ECG data from chest sensor
    ppg = data["signal"]["wrist"]["BVP"].squeeze()  # PPG (BVP) data from wrist sensor
    labels = data["label"]  # Labels from chest sensor


    unq, cnt = np.unique(labels, return_counts=True)
    print(f"We have {cnt} samples from {unq} original classes")

    # Preprocess PPG
    if args.ppgfm in ['papagei']:
        # # Optional plotting
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        # ax1.plot(ppg[:128], color='b')
        # ax1.set_title('Original PPG Data')
        # ax1.set_xlabel('Time (samples)')
        # ax1.set_ylabel('Amplitude')

        # Upsample PPG from 64Hz to 125Hz
        ppg_target_frequency = 125 #Hz
        num_samples = int(len(ppg) * ppg_target_frequency / SF_BVP)
        ppg_upsampled = signal.resample(ppg, num_samples)
        # Apply bandpass filter to PPG
        nyquist = 0.5 * ppg_target_frequency
        low = 0.5 / nyquist
        high = 12 / nyquist
        b, a = signal.cheby1(4, 0.5, [low, high], btype="band")
        ppg_filtered = signal.filtfilt(b, a, ppg_upsampled)
        ppg_data = zscore(ppg_filtered)

        # # Optional plotting
        # ax2.plot(ppg_data[:250], color='r')
        # ax2.set_title('Processed ECG Data')
        # ax2.set_xlabel('Time (samples)')
        # ax2.set_ylabel('Amplitude')
        # output_filename = 'ppg_plot.png'
        # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        # plt.close()
    else:
        print(f'{args.ppgfm} preprocessing not implemented!')

    # Preprocess ECG
    if args.ecgfm in ['ecgfm']:
        # # Optional plotting
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        # ax1.plot(ecg_data[0, 0, :], color='b')
        # ax1.set_title('Original ECG Data')
        # ax1.set_xlabel('Time (samples)')
        # ax1.set_ylabel('Amplitude')

        desired_sample_rate = 500 #Hz
        ecg_data = resample_ecgfm(ecg, SF_ECG, desired_sample_rate)
        mean = ecg_data.mean(axis=-1, keepdims=True)
        ecg_data = ecg_data - mean
        ecg_data, std = lead_std_divide(
            ecg_data,
            constant_lead_strategy='zero',
        )

        # # Optional plotting
        # ax2.plot(ecg_data[0, 0, :], color='r')
        # ax2.set_title('Processed ECG Data')
        # ax2.set_xlabel('Time (samples)')
        # ax2.set_ylabel('Amplitude')
        # output_filename = 'ecg_plot.png'
        # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        # plt.close()
    # HuBERT-ECG: We do not upsample like HuBERT-ECG to 500Hz
    elif args.ecgfm in ['hubertecg']:
        # # Optional plotting
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        # ax1.plot(ecg[:1400], color='b')
        # ax1.set_title('Original ECG Data')
        # ax1.set_xlabel('Time (samples)')
        # ax1.set_ylabel('Amplitude')

        original_frequency = 700 #Hz
        filter_frequency = 500 #Hz
        ecg_target_frequency = 100 #Hz
        ecg_data = resample_ecgfm(ecg, original_frequency, filter_frequency)
        order = int(0.3 * filter_frequency)
        ecg_data, _, _ = filter_signal(signal=ecg_data, ftype='FIR', band='bandpass',
                                    order=order, frequency=[0.05, 47], 
                                    sampling_rate=filter_frequency)
        ecg_data = resample_ecgfm(ecg_data, filter_frequency, ecg_target_frequency) # downsample to 100Hz
        mean = ecg_data.mean(axis=-1, keepdims=True)
        ecg_data = ecg_data - mean
        ecg_data, std = lead_std_divide(
            ecg_data,
            constant_lead_strategy='zero',
        )

        # # Optional plotting
        # ax2.plot(ecg_data[:200], color='r')
        # ax2.set_title('Processed ECG Data')
        # ax2.set_xlabel('Time (samples)')
        # ax2.set_ylabel('Amplitude')
        # output_filename = 'ecg_plot.png'
        # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        # plt.close()
    elif args.ecgfm in ['ecgdualnet']:
        # Optional plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        ax1.plot(ecg[:1400], color='b')
        ax1.set_title('Original ECG Data')
        ax1.set_xlabel('Time (samples)')
        ax1.set_ylabel('Amplitude')

        original_frequency = 700 #Hz
        filter_frequency = 500 #Hz
        ecg_target_frequency = 300 #Hz
        ecg_data = resample_ecgfm(ecg, original_frequency, filter_frequency)
        order = int(0.3 * filter_frequency)
        ecg_data, _, _ = filter_signal(signal=ecg_data, ftype='FIR', band='bandpass',
                                    order=order, frequency=[0.05, 47], 
                                    sampling_rate=filter_frequency)
        ecg_data = resample_ecgfm(ecg_data, filter_frequency, ecg_target_frequency) # downsample to 100Hz
        mean = ecg_data.mean(axis=-1, keepdims=True)
        ecg_data = ecg_data - mean
        ecg_data, std = lead_std_divide(
            ecg_data,
            constant_lead_strategy='zero',
        )

        # Optional plotting
        ax2.plot(ecg_data[:600], color='r')
        ax2.set_title('Processed ECG Data')
        ax2.set_xlabel('Time (samples)')
        ax2.set_ylabel('Amplitude')
        output_filename = 'ecg_plot_ecgdualnet.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f'{args.ecgfm} preprocessing not implemented!')
    
    # # Optional plotting
    # unique_labels = np.unique(label)
    # plt.figure(figsize=(15, 10))
    # for i, lbl in enumerate(unique_labels):
    #     indices = np.where(label == lbl)[0]
    #     example = ecg_data[indices[0], 0, :]  # Take only the first example for each label
    #     plt.subplot(len(unique_labels), 1, i + 1)  # Adjust subplot indexing
    #     plt.plot(example)
    #     plt.title(f'Label: {lbl}, Example: 1')
    #     plt.xlabel('Time')
    #     plt.ylabel('Amplitude')

    # output_filename = 'ecg_plot.png'
    # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    # plt.close()

    # Split into segments and save    
    ecg_segments, ecg_segment_labels = split_into_segments(ecg_data, np.round(signal.resample(labels, len(ecg_data))).astype(int), ecg_target_frequency, args.segment_length, args.step_size)
    ppg_segments, ppg_segment_labels = split_into_segments(ppg_data, np.round(signal.resample(labels, len(ppg_data))).astype(int), ppg_target_frequency, args.segment_length, args.step_size)
    assert ecg_segments.shape[0] == ppg_segments.shape[0], f"ECG and PPG segments do not match. ECG: {ecg_segments.shape[0]}, PPG: {ppg_segments.shape[0]}"
    assert ecg_segment_labels.shape[0] == ppg_segment_labels.shape[0]
    if not np.array_equal(ecg_segment_labels, ppg_segment_labels):
        raise ValueError("ECG and PPG labels do not match. Check segmentation logic.")
    segment_labels = ecg_segment_labels
    if args.num_classes == 2:
        mapping = {1: 0, 2: 1, 3: 0}
        segment_labels = np.array([mapping[label] for label in segment_labels])
    elif args.num_classes == 3:
        mapping = {1: 0, 2: 1, 3: 2}
        segment_labels = np.array([mapping[label] for label in segment_labels])

    for i in range(ecg_segments.shape[0]):
        # Save the segment
        file_name = f"subject_{str(subject_id).zfill(2)}_segment_{i}"
        ecg_segment = ecg_segments[i, :]
        label_segment = segment_labels[i]
        ppg_segment = ppg_segments[i, :]
        # Print how many segments is from each class using np.unique
        np.savez(os.path.join(args.output_dir, split, file_name), ECG=ecg_segment, PPG=ppg_segment, label=label_segment)
        
    unq, cnt = np.unique(segment_labels, return_counts=True)
    print(f"Split: {split}, File: {file}, We have {cnt} segments from {unq} our mapped classes")

if __name__ == "__main__":
    """
    WESAD dataset is downloaded from https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download
    """
    
    # Define configurations
    parser = argparse.ArgumentParser(description="Configuration for Processing")
    parser.add_argument("--wesad_path", type=str, default="/data/anonymous/WESAD", help="Path to the WESAD dataset")
    parser.add_argument("--splits", type=dict, default={"train": 5, "val": 2, "pair": 5, "test": 3}, help="Number of subjects for each split")
    parser.add_argument("--ecgfm", type=str, default='ecgdualnet', choices=['ecgfm', 'hubertecg', 'ecgdualnet'])
    parser.add_argument("--ppgfm", type=str, default='papagei', choices=['papagei'])
    parser.add_argument("--segment_length", type=int, default=60, help='Length of each segment in seconds')
    parser.add_argument("--step_size", type=int, default=5, help='Step size between adjacent segments in seconds')
    parser.add_argument("--num_classes", type=int, default=3, help='Do 3 class baseline vs stress vs amusement or 2 class stress vs non-stress')
    args = parser.parse_args()

    # Define output directory based on configurations
    args.output_dir = os.path.join(args.wesad_path, f'processed_ppgfm{args.ppgfm}_ecgfm{args.ecgfm}_{args.segment_length}s_{args.step_size}sstep_{args.num_classes}classes')

    # Setup the output folders
    folders = args.splits.keys()
    for folder in folders:
        if not os.path.exists(os.path.join(args.output_dir, folder)):
            os.makedirs(os.path.join(args.output_dir, folder))

    # Initialize the dataset_summary.txt and redirect all prints to the text file, will be restored to stdout later
    output_file_path = os.path.join(args.output_dir, "dataset_summary.txt")
    sys.stdout = open(output_file_path, 'w', buffering=1)

    # Create splits by subjects
    files = os.listdir(os.path.join(args.wesad_path))
    files = [file for file in files if re.match(r'^S\d+', file)] # filter to only look at relevant files
    random.shuffle(files)
    split_files = {}
    cnt = 0
    for split,split_size in args.splits.items():
        split_files[split] = files[cnt:cnt+split_size]
        cnt += split_size
        print(f'For split {split}, we have subjects {split_files[split]}')

    # Preprocess, Split, Save
    for split, files in split_files.items():
        for file in tqdm(files, desc=f"Processing {split}", unit="file"):
            preprocess_split_save(args, split, file)

    # Print arguments used
    print(args)

    # Restore original stdout at the end of the script
    sys.stdout.close()
    sys.stdout = sys.__stdout__













