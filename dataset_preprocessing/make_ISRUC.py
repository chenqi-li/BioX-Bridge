import os
import random
import pickle

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

random.seed(6)

# ISRUC Preprocessing
# For EEG:
# Split each session into 30s segments.

# Running this file: 
# Change the settings in args
# (BioX-Bridge) [anonymous@server /users/anonymous/BioX-Bridge/dataset_preprocessing]$ python make_ISRUC.py


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

# Preprocess, Split, Save
def preprocess_split_save(args, split, file):
    # Load data and calculate basic variables
    subject_id = int(file.replace("subject", "").replace(".mat", ""))
    psg = scipy.io.loadmat(os.path.join(args.isruc_path, 'ExtractedChannels', file))
    eeg_channels = ['F3_A2', 'C3_A2', 'O1_A2', 'F4_A1', 'C4_A1', 'O2_A1']
    eeg_data = []
    for c in eeg_channels:
        eeg_data.append(np.expand_dims(psg[c],1))
    eeg_data = np.concatenate(eeg_data, axis=1)
    ecg_data = np.expand_dims(psg['X2'],1)
    n_epoch = eeg_data.shape[0]

    # Preprocess EEG
    if args.eegfm in ['labram']:
        eeg_data = eeg_data # No need to convert unit to 0.1mV for range [-1, 1] since that is done in train_one_epoch of LaBraM
    else:
        print(f'{args.eegfm} preprocessing not implemented!')

    # Preprocess ECG
    # ECG-FM: We follow their preprocessing code exactly
    if args.ecgfm in ['ecgfm']:
        # # Optional plotting
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        # ax1.plot(ecg_data[0, 0, :], color='b')
        # ax1.set_title('Original ECG Data')
        # ax1.set_xlabel('Time (samples)')
        # ax1.set_ylabel('Amplitude')

        curr_sample_rate = 200 #Hz
        desired_sample_rate = 500 #Hz
        ecg_data = resample_ecgfm(ecg_data, curr_sample_rate, desired_sample_rate)
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
        # ax1.plot(ecg_data[0, 0, :500], color='b')
        # ax1.set_title('Original ECG Data')
        # ax1.set_xlabel('Time (samples)')
        # ax1.set_ylabel('Amplitude')

        original_frequency = 200 #Hz
        filter_frequency = 500 #Hz
        target_frequency = 100 #Hz
        ecg_data = resample_ecgfm(ecg_data, original_frequency, filter_frequency)
        order = int(0.3 * filter_frequency)
        ecg_data, _, _ = filter_signal(signal=ecg_data, ftype='FIR', band='bandpass',
                                    order=order, frequency=[0.05, 47], 
                                    sampling_rate=filter_frequency)
        ecg_data = resample_ecgfm(ecg_data, filter_frequency, target_frequency) # downsample to 100Hz
        mean = ecg_data.mean(axis=-1, keepdims=True)
        ecg_data = ecg_data - mean
        ecg_data, std = lead_std_divide(
            ecg_data,
            constant_lead_strategy='zero',
        )

        # # Optional plotting
        # ax2.plot(ecg_data[0, 0, :250], color='r')
        # ax2.set_title('Processed ECG Data')
        # ax2.set_xlabel('Time (samples)')
        # ax2.set_ylabel('Amplitude')
        # output_filename = 'ecg_plot.png'
        # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        # plt.close()
    else:
        print(f'{args.ecgfm} preprocessing not implemented!')

    # Load label
    label = []
    with open(os.path.join(args.isruc_path, 'RawData', f'{subject_id}/{subject_id}_1.txt')) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s == '' or s == '\n':
                break
    label = np.array(label[:-30]) # ignore last 30 segments, because noise, see ISRUC website under Extracted Channels tab
    if args.num_classes == 5:
        label[label==5] = 4  # make 4 correspond to REM, in ISRUC, 0-Wake, 1-N1, 2-N2, 3-N3, 5-REM because N3 used to be further split into N3 and N4
    elif args.num_classes == 2:
        label[label==2] = 1
        label[label==3] = 1
        label[label==5] = 1 # 0-wake, (1-N1, 2-N2, 3-N3, 5-REM) -> 1-sleep

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

    # Split and Save
    for i in range(n_epoch):
        # Save the segment
        file_name = f"subject_{str(subject_id).zfill(3)}_segment_{i}"
        eeg_segment = eeg_data[i, :, :]
        label_segment = label[i]
        ecg_segment = ecg_data[i, :, :]
        np.savez(os.path.join(args.output_dir, split, file_name), EEG=eeg_segment, ECG=ecg_segment, label=label_segment)


if __name__ == "__main__":
    """
    ISRUC dataset is downloaded using /data/anonymous/LaBraM-anonymous/download_ISRUC_S1.sh
    """
    
    # Define configurations
    parser = argparse.ArgumentParser(description="Configuration for Processing")
    parser.add_argument("--isruc_path", type=str, default="/data/anonymous/ISRUC_S1", help="Path to the ISRUC dataset")
    parser.add_argument("--splits", type=dict, default={"train": 33, "val": 11, "pair": 33, "test": 22}, help="Number of subjects for each split")
    parser.add_argument("--eeg_sampling_freq", type=int, default=200, help="EEG sampling frequency in Hz")
    parser.add_argument("--ecgfm", type=str, default='hubertecg', choices=['ecgfm', 'hubertecg'])
    parser.add_argument("--eegfm", type=str, default='labram', choices=['labram'])
    parser.add_argument("--num_classes", type=int, default=2, help='Do 5 class sleep staging or 2 class sleep/wake detection')
    args = parser.parse_args()

    # Define output directory based on configurations
    args.output_dir = os.path.join(args.isruc_path, f'processed_ecgfm{args.ecgfm}_eegfm{args.eegfm}_{args.num_classes}classes')

    # Setup the output folders
    folders = args.splits.keys()
    for folder in folders:
        if not os.path.exists(os.path.join(args.output_dir, folder)):
            os.makedirs(os.path.join(args.output_dir, folder))

    # Initialize the dataset_summary.txt and redirect all prints to the text file, will be restored to stdout later
    output_file_path = os.path.join(args.output_dir, "dataset_summary.txt")
    sys.stdout = open(output_file_path, 'w', buffering=1)

    # Create splits by subjects
    files = os.listdir(os.path.join(args.isruc_path, 'ExtractedChannels'))
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
