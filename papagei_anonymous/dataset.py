# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch 
import os
import pandas as pd
import numpy as np
import augmentations
import matplotlib.pyplot as plt
import joblib
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch_ecg._preprocessors import Normalize
from functools import lru_cache


class PPGDataset(Dataset):
    """
    Patient level positive pair selection
    """
    def __init__(self, df, path, case_name, label_name, fs, normalization=True, simclr=True, transform=None):
        """
        Args:
            df (pandas.DataFrame): Dataframe consisting of filename and label name
            path (string): directory path to vitaldb pickle files
            label_name (string): label name to extract from df
            waveform (string): waveform name to extract from pickle
            normalization (boolean): whether to normalize signal or not
            transform (torchvision.transforms.Compose): Data augmentation or transforms for the signal
        """
        self.filenames = np.unique(df.loc[:, case_name].values)
        self.dict_case = df.groupby(case_name)['segments'].apply(list).to_dict()
        self.path = path
        self.fs = fs
        self.normalization = normalization
        self.transform = transform 
        self.simclr = simclr

        # patient level labels
        df = df.drop_duplicates(subset=[case_name])
        self.labels = [df[df[case_name] == f][label_name].values[0] for f in self.filenames]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Choose a case file
        case_file = self.filenames[idx]
        # Randomly select a segment id
        # segment_list = os.listdir(os.path.join(self.path, case_file))
        # segment_idx = np.random.choice(np.arange(0, len(segment_list)), size=1)[0]
        # signal = joblib.load(os.path.join(self.path, case_file, str(segment_list[segment_idx])))

        segment_list = self.dict_case[case_file]
        segment_idx = np.random.choice(np.arange(0, len(segment_list)), size=1)[0]
        signal_segment = os.path.join(self.path, case_file, str(segment_list[segment_idx]))
        if not signal_segment.endswith(".p"):
            signal_segment = signal_segment + ".p"
        signal = joblib.load(signal_segment)
        label = self.labels[idx]

        if self.normalization:
            # signal = self.normalize(signal)
            norm = Normalize(method='z-score')
            signal, _ = norm.apply(signal, fs=self.fs)

        if signal.ndim != 2:
            signal = np.expand_dims(signal, axis=0)

        if self.simclr:
            # positive pair views for SimCLR
            signal_view1 = torch.Tensor(self.transform(signal))
            signal_view2 = torch.Tensor(self.transform(signal))
            
            return [signal_view1.squeeze(dim=1), signal_view2.squeeze(dim=1)], label
        else:
            if self.transform:
                signal = self.transform(signal)
            return signal, label

class PPGDatasetLabelsArray(Dataset):

    def __init__(self, df, fs_target, normalization=True, simclr=True, transform=None, bins_svri=8, bins_skewness=5, binary_ipa=False):
        """
        Args:
            df (pandas.DataFrame): Dataframe consisting of filename and label name
            path (string): directory path to vitaldb pickle files
            label_name (string): label name to extract from df
            waveform (string): waveform name to extract from pickle
            normalization (boolean): whether to normalize signal or not
            transform (torchvision.transforms.Compose): Data augmentation or transforms for the signal
        """

        self.normalization = normalization
        self.transform = transform 
        self.simclr = simclr
            
        paths = df['path'].values
        cases = df['case_id'].values
        segments = df['segments'].values
        self.fs = df['fs'].values
        self.fs_target = fs_target
        self.filenames = [f"{paths[i]}{cases[i]}/{segments[i]}" for i in range(len(cases))]
        self.resample_500 = augmentations.ResampleSignal(fs_original=500, fs_target=self.fs_target)
        self.resample_256 = augmentations.ResampleSignal(fs_original=256, fs_target=self.fs_target)

        svri = np.digitize(df['svri'].values, bins=self.bin_data(df['svri'].values, bins_svri)) - 1

        if bins_skewness == 0:
            skewness = df['skewness'].values
        else:
            skewness = np.digitize(df['skewness'].values, bins=self.bin_data(df['skewness'].values, bins_skewness)) - 1
        ipa = df['ipa'].values

        if binary_ipa:
            ipa = np.where(ipa == 0, 0, 1)
        
        self.labels = np.column_stack((svri, skewness, ipa))

    def __len__(self):
        return len(self.labels)

    @lru_cache(maxsize=1024) 
    def load_signal(self, idx):
        signal = joblib.load(self.filenames[idx])
        if self.normalization:
            norm = Normalize(method='z-score')
            signal, _ = norm.apply(signal, fs=self.fs[idx])
        if signal.ndim != 2:
            signal = np.expand_dims(signal, axis=0)
        return signal
        
    def bin_data(self, values, num_buckets):
        min_value = values.min()
        max_value = values.max()
        
        bucket_width = (max_value - min_value) / num_buckets
        buckets = np.arange(min_value, max_value + bucket_width, bucket_width)
        return buckets
        
    def __getitem__(self, idx):
        signal = self.load_signal(idx)
        if "vital" in self.filenames[idx]:
            signal = self.resample_500(signal)
        if "mesa" in self.filenames[idx]:
            signal = self.resample_256(signal)
        signal = torch.Tensor(self.transform(signal))
        return signal.squeeze(dim=1), self.labels[idx]
        
class PPGDatasetVanillaSimCLR(Dataset):

    def __init__(self, df, fs_target, normalization=True, simclr=True, transform=None):
        """
        Args:
            df (pandas.DataFrame): Dataframe consisting of filename and label name
            path (string): directory path to vitaldb pickle files
            label_name (string): label name to extract from df
            waveform (string): waveform name to extract from pickle
            normalization (boolean): whether to normalize signal or not
            transform (torchvision.transforms.Compose): Data augmentation or transforms for the signal
        """

        self.normalization = normalization
        self.transform = transform 
        self.simclr = simclr
            
        paths = df['path'].values
        cases = df['case_id'].values
        segments = df['segments'].values
        self.fs = df['fs'].values
        self.fs_target = fs_target
        self.filenames = [f"{paths[i]}{cases[i]}/{segments[i]}" for i in range(len(cases))]
        self.resample_500 = augmentations.ResampleSignal(fs_original=500, fs_target=self.fs_target)
        self.resample_256 = augmentations.ResampleSignal(fs_original=256, fs_target=self.fs_target)

    def __len__(self):
        return len(self.filenames)

    @lru_cache(maxsize=1024) 
    def load_signal(self, idx):
        signal = joblib.load(self.filenames[idx])
        if self.normalization:
            norm = Normalize(method='z-score')
            signal, _ = norm.apply(signal, fs=self.fs[idx])
        if signal.ndim != 2:
            signal = np.expand_dims(signal, axis=0)
        return signal
        
    def __getitem__(self, idx):
        signal = self.load_signal(idx)
        if "vital" in self.filenames[idx]:
            signal = self.resample_500(signal)
        if "mesa" in self.filenames[idx]:
            signal = self.resample_256(signal)
        signal_v1 = torch.Tensor(self.transform(signal))
        signal_v2 = torch.Tensor(self.transform(signal))
        return signal_v1.squeeze(dim=1), signal_v2.squeeze(dim=1)

def generate_dataset(CustomDataset, df, path, case_name, label_name, fs, normalization, simclr, transform):

    """
    Generates a dataset based on custom class

    Args:
        CustomDataset (torch.utils.data.Dataset): Custom torch dataset class
        df (pandas.Dataframe): Dataframe with filenames and labels
        case_name (string): column name for filenames
        label_name (string): column name for labels
        fs (int): original sampling frequency 
        normalization (boolean): whether to normalize or not
        simclr (boolean): simclr style outputs or not
        transform (torchvision.transforms.Compose): transforms to apply to the signals

    Returns:
        dataset (torch.utils.data.Dataset): A dataset object to pass to dataloader
    """
    
    dataset = CustomDataset(df=df,
                        path=path, 
                        case_name=case_name,
                        label_name=label_name,
                        fs=fs,
                        normalization=normalization,
                        simclr=simclr,
                        transform=transform)
    return dataset

def generate_dataloader(dataset, batch_size, shuffle, num_workers, distributed=False):
    """
    Generates a dataloader based for the dataset 

    Note: shuffle must be False for distributed training

    Args:
        dataset (torch.utils.data.Dataset): A dataset object to pass to dataloader
        batch_size (int): batch size for training
        shuffle (boolean): whether to shuffle or not
        num_workers (int): no. of workers for loading 
        distributed (boolean): whether training is going to distributed or not.

    Returns:
        dataloader (torch.utils.data.DataLoader): A dataloader object for training
    """

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        sampler=sampler,
                        persistent_workers=True,
                        drop_last=True)
    else:
        sampler = None
        dataloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        sampler=sampler,
                        persistent_workers=True,
                        drop_last=True)
        

    return dataloader

def load_dataset_obj(dataset_name, CustomDataset, label_name, fs_target, normalization, simclr, transform):
    """
    Load dataset objects for the different pretraining datasets

    Args:
        dataset_name (string): The dataset name, choose from mesa, vital, or mimic.
        CustomDataset (torch.utils.data.Dataset): Custom torch dataset class
        label_name (string): column name for labels
        fs_target (int): target sampling frequency for resampling
        normalization (boolean): whether to normalize or not
        simclr (boolean): simclr style outputs or not
        transform (torchvision.transforms.Compose): transforms to apply to the signals

    Returns:
        train_dataset (torch.utils.data.Dataset): train dataset object
        val_dataset (torch.utils.data.Dataset): val dataset object
        test_dataset (torch.utils.data.Dataset): test dataset object

    """
    if dataset_name == "mesa":
        case_name = "mesaid"
        path = "../data/mesa/mesappg/"
        fs=256
        usecols = [case_name, "segments", "age", label_name]

        df_train = pd.read_csv("../data/mesa/train_clean.csv", usecols=usecols)
        df_val = pd.read_csv("../data/mesa/val_clean.csv", usecols=usecols)
        df_test = pd.read_csv("../data/mesa/test_clean.csv", usecols=usecols)

        df_train.loc[:, 'mesaid'] = df_train.mesaid.apply(lambda x: str(x).zfill(4))
        df_val.loc[:, 'mesaid'] = df_val.mesaid.apply(lambda x: str(x).zfill(4))
        df_test.loc[:, 'mesaid'] = df_test.mesaid.apply(lambda x: str(x).zfill(4))
        


    if dataset_name == "vital":
        path = "../data/vitaldbppg/"
        case_name = "caseid"
        fs=500
        usecols = [case_name, "segments", "age"]

        df_train = pd.read_csv("../data/vital/train_clean.csv", usecols=usecols)
        df_val = pd.read_csv("../data/vital/val_clean.csv", usecols=usecols)
        df_test = pd.read_csv("../data/vital/test_clean.csv", usecols=usecols)

        df_train.loc[:, 'caseid'] = df_train.caseid.apply(lambda x: str(x).zfill(4))
        df_val.loc[:, 'caseid'] = df_val.caseid.apply(lambda x: str(x).zfill(4))
        df_test.loc[:, 'caseid'] = df_test.caseid.apply(lambda x: str(x).zfill(4))
        


    if dataset_name == "mimic":
        case_name = "SUBJECT_ID"
        path = "../data/mimic/ppg_filt/" # Twice filtered mimic data for training
        fs=125
        usecols = [case_name, "segments", "age"]

        df_train = pd.read_csv("../data/mimic/train_clean.csv", usecols=usecols)
        df_val = pd.read_csv("../data/mimic/val_clean.csv", usecols=usecols)
        df_test = pd.read_csv("../data/mimic/test_clean.csv", usecols=usecols)


    train_transform = transform[:]
    train_transform.insert(0, augmentations.ResampleSignal(fs, fs_target))
    train_transform = transforms.Compose(train_transform)

    vt_transforms = transforms.Compose([augmentations.ResampleSignal(fs, fs_target),
                                       transforms.ToTensor()])
    
    train_dataset = generate_dataset(CustomDataset=CustomDataset,
                                    df=df_train, 
                                    path=path, 
                                    case_name=case_name, 
                                    label_name=label_name, 
                                    fs=fs, 
                                    normalization=normalization, 
                                    simclr=simclr, 
                                    transform=train_transform)
    
    val_dataset = generate_dataset(CustomDataset=CustomDataset,
                                df=df_val, 
                                path=path, 
                                case_name=case_name, 
                                label_name=label_name, 
                                fs=fs, 
                                normalization=normalization, 
                                simclr=simclr, 
                                transform=vt_transforms)
    
    test_dataset = generate_dataset(CustomDataset=CustomDataset,
                                    df=df_test, 
                                    path=path, 
                                    case_name=case_name, 
                                    label_name=label_name, 
                                    fs=fs, 
                                    normalization=normalization, 
                                    simclr=simclr, 
                                    transform=vt_transforms)

    return train_dataset, val_dataset, test_dataset

def load_dataloader_obj(train_dataset, val_dataset, test_dataset, batch_size, shuffle, num_workers=2, distributed=False):

    """
    Generate dataloaders using the given dataset classes.

    Args:
        train_dataset (torch.utils.data.Dataset): train dataset object
        val_dataset (torch.utils.data.Dataset): val dataset object
        test_dataset (torch.utils.data.Dataset): test dataset object
        batch_size (int): batch size for training
        shuffle (boolean): whether to shuffle or not
        num_workers (int): no. of workers for loading 
        distributed (boolean): whether training is going to distributed or not.

    Returns:
        train_dataloader (torch.utils.data.DataLoader): training dataloader object 
        val_dataloader (torch.utils.data.DataLoader): val dataloader object 
        test_dataloader (torch.utils.data.DataLoader): test dataloader object
    """
    
    train_dataloader = generate_dataloader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=num_workers,
                                          distributed=distributed)
    
    val_dataloader = generate_dataloader(dataset=val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      distributed=distributed)
    
    test_dataloader = generate_dataloader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      distributed=distributed)
    
    return train_dataloader, val_dataloader, test_dataloader

def dataset_selector(key, CustomDataset, label_name, fs_target, simclr_transform, batch_size, shuffle, distributed):
    """
    Selects dataset for training by generating datasets and dataloaders

    Args:
        key (string): dataset key; Choose from vital, mesa, mimic, vital_mesa, vital_mimic, mesa_mimic, vital_mesa_mimic
        fs_target (int): target resampling frequency
        simclr_transform (list): This is a list of transforms of torch.nn.Module (not a transforms.Compose)
        batch_size (int): loading batch size
        shuffle (boolean): shuffle dataset or not
        distributed (boolean): dataloaders are boolean or not

    Returns:
        train_dataloader (torch.utils.data.DataLoader): dataloader for training simclr style
        val_dataloader (torch.utils.data.DataLoader): dataloader for validation
        test_dataloader (torch.utils.data.DataLoader): dataloader for testing

    """
    
    mesa_train_dataset, mesa_val_dataset, mesa_test_dataset = load_dataset_obj(dataset_name="mesa",
                                                                             CustomDataset=CustomDataset,
                                                                             label_name=label_name, 
                                                                             fs_target=fs_target,
                                                                             normalization=True, 
                                                                             simclr=True, 
                                                                             transform=simclr_transform)
    
    vital_train_dataset, vital_val_dataset, vital_test_dataset = load_dataset_obj(dataset_name="vital",
                                                                             CustomDataset=CustomDataset,
                                                                             label_name=label_name, 
                                                                             fs_target=fs_target,
                                                                             normalization=True, 
                                                                             simclr=True, 
                                                                             transform=simclr_transform)
    
    mimic_train_dataset, mimic_val_dataset, mimic_test_dataset = load_dataset_obj(dataset_name="mimic",
                                                                             CustomDataset=CustomDataset,
                                                                             label_name=label_name, 
                                                                             fs_target=fs_target,
                                                                             normalization=True, 
                                                                             simclr=True, 
                                                                             transform=simclr_transform)

    if key == "vital":
        train_dataset = vital_train_dataset
        val_dataset = vital_val_dataset
        test_dataset = vital_test_dataset

    if key == "mesa":
        train_dataset = mesa_train_dataset
        val_dataset = mesa_val_dataset
        test_dataset = mesa_test_dataset

    if key == "mimic":
        train_dataset = mimic_train_dataset
        val_dataset = mimic_val_dataset
        test_dataset = mimic_test_dataset

    if key == "vital_mesa":
        train_dataset = ConcatDataset(datasets=[mesa_train_dataset, vital_train_dataset])
        val_dataset = ConcatDataset(datasets=[mesa_val_dataset, vital_val_dataset])
        test_dataset = ConcatDataset(datasets=[mesa_test_dataset, vital_test_dataset])

    if key == "vital_mimic":
        train_dataset = ConcatDataset(datasets=[vital_train_dataset, mimic_train_dataset])
        val_dataset = ConcatDataset(datasets=[vital_val_dataset, mimic_val_dataset])
        test_dataset = ConcatDataset(datasets=[vital_test_dataset, mimic_test_dataset])

    if key == "mesa_mimic":
        train_dataset = ConcatDataset(datasets=[mesa_train_dataset, mimic_train_dataset])
        val_dataset = ConcatDataset(datasets=[mesa_val_dataset, mimic_val_dataset])
        test_dataset = ConcatDataset(datasets=[mesa_test_dataset, mimic_test_dataset])

    if key == "vital_mesa_mimic":
        train_dataset = ConcatDataset(datasets=[mesa_train_dataset, vital_train_dataset, mimic_train_dataset])
        val_dataset = ConcatDataset(datasets=[mesa_val_dataset, vital_val_dataset, mimic_val_dataset])
        test_dataset = ConcatDataset(datasets=[mesa_test_dataset, vital_test_dataset, mimic_test_dataset])

    train_dataloader, val_dataloader, test_dataloader = load_dataloader_obj(train_dataset=train_dataset,
                                                                       val_dataset=val_dataset,
                                                                       test_dataset=test_dataset,
                                                                       batch_size=batch_size,
                                                                       shuffle=shuffle,
                                                                       distributed=distributed)
    return train_dataloader, val_dataloader, test_dataloader
