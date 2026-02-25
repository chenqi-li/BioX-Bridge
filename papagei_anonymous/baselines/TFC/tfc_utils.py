import torch
import torch.nn.functional as F
import numpy as np
import sys
import pandas as pd
sys.path.append("../../../papagei-foundation-model/")
import torch.fft as fft
import torch.nn as nn
import torch.optim as optim
import augmentations
import joblib

from tqdm import tqdm
from training_pospair import harmonize_datasets
from transforms import DataTransform_FD, DataTransform_TD
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from functools import lru_cache
from torch_ecg._preprocessors import Normalize
from models.resnet import BasicBlock, MyConv1dPadSame, MyMaxPool1dPadSame


class TFCDataset(Dataset):

    def __init__(self, df, fs_target, config, normalization=True, simclr=True, transform=None):
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
        self.config = config
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
            
        x_data = torch.from_numpy(signal)
        x_data_f = fft.fft(x_data).abs()
        aug1 = DataTransform_TD(x_data, self.config)
        aug1_f = DataTransform_FD(x_data_f, self.config)

        return x_data, aug1, x_data_f, aug1_f


class NTXentLoss_poly(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss_poly, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        CE = self.criterion(logits, labels)

        onehot_label = torch.cat((torch.ones(2 * self.batch_size, 1),torch.zeros(2 * self.batch_size, negatives.shape[-1])),dim=-1).to(self.device).long()
        # Add poly loss
        pt = torch.mean(onehot_label* torch.nn.functional.softmax(logits,dim=-1))

        epsilon = self.batch_size
        # loss = CE/ (2 * self.batch_size) + epsilon*(1-pt) # replace 1 by 1/self.batch_size
        loss = CE / (2 * self.batch_size) + epsilon * (1/self.batch_size - pt)
        # loss = CE / (2 * self.batch_size)

        return loss


