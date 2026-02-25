import io
import os
import math
import time
import json
import glob
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict
from pathlib import Path
import argparse
import torch
import torch.distributed as dist
from torch import inf
import h5py
from tensorboardX import SummaryWriter
import pickle
from scipy.signal import resample
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from LaBraM_anonymous.utils import MetricLogger, get_metrics
from functools import partial
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from copy import deepcopy

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))





# Data loader returns data batches of shape [batch_size, in_channels, sequence_length]
class ISRUCLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, order, sampling_rate=200, percentage=100):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.order = order
        if percentage > 0 and percentage < 100:
            import random
            total = len(files)
            num_samples = int((percentage / 100.0) * total)
            rng = random.Random(0) # shuffle with fixed seed to ensure consistency across seeds
            shuffled_files = files.copy()
            rng.shuffle(shuffled_files)
            self.files = shuffled_files[:num_samples]
            print(f'Using {percentage} of dataset, which is {num_samples} out of {total} samples.')
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = np.load(os.path.join(self.root, self.files[index]))
        EEG = sample["EEG"]
        ECG = sample["ECG"]
        label = sample["label"]
        EEG = torch.FloatTensor(EEG)
        ECG = torch.FloatTensor(ECG)
        
        if self.order == 'eegecglabel':
            return EEG, ECG, label
        elif self.order == 'ecgeeglabel':
            return ECG, EEG, label

# Data loader returns data batches of shape [batch_size, in_channels, sequence_length]
class WESADLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, order, percentage=100):
        self.root = root
        self.files = files
        self.order = order
        if percentage > 0 and percentage < 100:
            import random
            total = len(files)
            num_samples = int((percentage / 100.0) * total)
            rng = random.Random(0) # shuffle with fixed seed to ensure consistency across seeds
            shuffled_files = files.copy()
            rng.shuffle(shuffled_files)
            self.files = shuffled_files[:num_samples]
            print(f'Using {percentage} of dataset, which is {num_samples} out of {total} samples.')
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = np.load(os.path.join(self.root, self.files[index]))
        PPG = sample["PPG"]
        ECG = sample["ECG"]
        label = sample["label"]
        PPG = torch.FloatTensor(PPG).reshape(1, -1)
        ECG = torch.FloatTensor(ECG).reshape(1, -1)
        if self.order == 'ppgecglabel':
            return PPG, ECG, label
        elif self.order == 'ecgppglabel':
            return ECG, PPG, label

# Data loader returns data batches of shape [batch_size, in_channels, sequence_length]
class FOGLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, order, percentage=100):
        self.root = root
        self.files = files
        self.order = order
        if percentage > 0 and percentage < 100:
            import random
            total = len(files)
            num_samples = int((percentage / 100.0) * total)
            rng = random.Random(0) # shuffle with fixed seed to ensure consistency across seeds
            shuffled_files = files.copy()
            rng.shuffle(shuffled_files)
            self.files = shuffled_files[:num_samples]
            print(f'Using {percentage} of dataset, which is {num_samples} out of {total} samples.')
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = np.load(os.path.join(self.root, self.files[index]))
        EEG = sample["EEG"]
        EMG = sample["EMG"]
        label = sample["label"]
        EEG = torch.FloatTensor(EEG)
        EMG = torch.FloatTensor(EMG)
        
        if self.order == 'eegemglabel':
            return EEG, EMG, label
        elif self.order == 'emgeeglabel':
            return EMG, EEG, label

# # Data preparation LaBraM
def labram_data_prepare(args, device, data):
    data = data.float().to(device, non_blocking=True) / 100
    data = rearrange(data, 'B N (A T) -> B N A T', T=200)
    return data

# Data preparation HuBERTECG
def hubertecg_data_prepare(args, device, data):
    if args.ecg_channel_expansion in ['zero_pad']:
        data = F.pad(data, (0,0,0,11,0,0), mode='constant', value=0)
    elif args.ecg_channel_expansion in ['repeat']:
        data = data.repeat(1, 12, 1)
    elif args.ecg_channel_expansion in ['none']:
        pass
    curr_batch_size = data.shape[0]
    data = data.reshape(curr_batch_size, -1)
    return data
hubertecgb_data_prepare = hubertecgl_data_prepare = hubertecg_data_prepare

# Data preparation ECGFM
def ecgfm_data_prepare(args, device, data):
    if args.ecg_channel_expansion in ['zero_pad']:
        data = F.pad(data, (0,0,0,11,0,0), mode='constant', value=0)
    elif args.ecg_channel_expansion in ['repeat']:
        data = data.repeat(1, 12, 1)
    return data

# Data preparation ecgdualnet
from torchaudio.transforms import Spectrogram                   
def ecgdualnet_data_prepare(args, device, data):
    
    spectrogram_module = Spectrogram(n_fft=64, win_length=64, hop_length=64 // 2, power=1, normalized=True).to(device)
    spectrogram = spectrogram_module(data.squeeze(1))
    spectrogram = torch.log(spectrogram.abs().clamp(min=1e-08))
    spectrogram = spectrogram.permute(0, 2, 1)
    spectrogram = spectrogram.unsqueeze(dim=1).float()

    data = data.unfold(dimension=-1, size=256, step=256-32)
    data = data.squeeze(1)
    
    return (data, spectrogram)

# Data preparation PaPaGei
def papagei_data_prepare(args, device, data):
    return data

def normwear_data_prepare(args, device, data):
    return data





def hubertecg_forward(fm, data, messenger):
    # model expects data of shape [batch_size, in_channels*sequence_length] # ECG in_channels=12
    _ = fm(data, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True)
    return messenger
hubertecgb_forward = hubertecgl_forward = hubertecg_forward

def ecgfm_forward(fm, data, messenger):
    # model expects data of shape [batch_size, in_channels, sequence_length] # ECG in_channels=12
    _ = fm(source=data)
    return messenger

def labram_forward(fm, data, messenger):
    # model expects data of shape [batch_size, in_channels, num_seconds, 200] # Each token is 1 second, so 200
    with torch.cuda.amp.autocast():
        _ = fm(data, input_chans=None, norm_output=False)
    return messenger

def papagei_forward(fm, data, messenger):
    # expects data of shape [batch_size, in_channels, sequence_length] # PPG in_channels=1
    _ = fm(data)
    return messenger        

def ecgdualnet_forward(fm, data, messenger):
    # model expects data of shape [batch_size, in_channels*sequence_length] # ECG in_channels=12
    _ = fm(ecg_lead=data[0], spectrogram=data[1])
    return messenger

def normwear_forward(fm, data, messenger):
    # model expects data of shape [batch_size, nvar, 3, L, n_mels]
    _ = fm.get_embedding(data, sampling_rate=65) # bn, nvar, P, E
    return messenger




def fm_forward(fm_old, fm_new, data_old, data_new, messenger, args):
    # Compute features from old modality if fm_old and data_old is provided
    if fm_old is None and data_old is None:
        fm_old_input = None
        fm_old_lastlayer_output = None
    else:
        messenger = eval(f"{args.fm_old}_forward")(fm=fm_old, data=data_old, messenger=messenger)
        fm_old_input = messenger['fm_input'][0]
        fm_old_lastlayer_output = messenger['fm_lastlayer_output']
    
    # Compute features from new modality if fm_new and data_new is provided
    if fm_new is None and data_new is None:
        fm_new_output = None
    else:
        messenger = eval(f"{args.fm_new}_forward")(fm=fm_new, data=data_new, messenger=messenger)
        if 'fm_output' in messenger:
            fm_new_output = messenger['fm_output']
        else:
            fm_new_output = None
    
    # Reshape for normwear
    if args.fm_old in ['normwear'] and data_old is not None:
        BN, P, E = fm_old_lastlayer_output.shape
        fm_old_lastlayer_output = torch.reshape(fm_old_lastlayer_output,(data_old.shape[0], -1, P, E))
        BN, P, E = fm_old_input.shape
        fm_old_input = torch.reshape(fm_old_input,(data_old.shape[0], -1, P, E))
    elif args.fm_new in ['normwear'] and data_new is not None:
        BN, P, E = fm_new_output.shape
        fm_new_output = torch.reshape(fm_new_output,(data_new.shape[0], -1, P, E))

    return fm_old_input, fm_new_output, fm_old_lastlayer_output

# This structure is copied from LaBraM engine_for_finetuning evaluate
@torch.no_grad()
def evaluate_bridge(args, data_loader, bridge_criterion, fm_old, fm_new, bridge_model, messenger, device, ch_names, metrics, is_binary, linear_prober=None, resnet_model=None, additional_prober=None):

    # Initialize the loss
    if is_binary:
        logits_criterion = torch.nn.BCEWithLogitsLoss()
    else:
        logits_criterion = torch.nn.CrossEntropyLoss()

    # Initialize metric logger
    metric_logger = MetricLogger(delimiter="  ")

    # Intialize variables for recording statistics
    pred = []
    true = []
    
    # Loop through dataloader
    for step, batch in enumerate(data_loader):
        # Prepare the data
        current_batch_size = batch[0].shape[0]
        data_old = batch[0].to(device)
        data_new = batch[1].to(device)
        target = batch[-1]
        target = target.to(device, non_blocking=True)
        data_new = eval(f"{args.fm_new}_data_prepare")(args=args, device=device, data=data_new)
        data_old = eval(f"{args.fm_old}_data_prepare")(args=args, device=device, data=data_old)

        # Forward inference from data_new to bridge_output
        if args.mode in ['train', 'evaluate']:
            with torch.no_grad(): # Forward for FMs
                fm_old_input, fm_new_output, fm_old_lastlayer_output = fm_forward(fm_old=fm_old, fm_new=fm_new, data_old=data_old, data_new=data_new, messenger=messenger, args=args)            

            # Compute bridge features from new features
            if args.bridge_input_dim_reduction in ['mean']:
                if step == 0: # only calculate the dimension to reduce at the first step
                    bridge_dims_to_reduce = [i for i, dim in enumerate(fm_new_output.shape) if dim != current_batch_size and dim != args.bridge_input_dim]
                fm_new_output = torch.mean(fm_new_output, dim=tuple(bridge_dims_to_reduce))
            bridge_output = bridge_model(fm_new_output.view(current_batch_size,-1))
            
            # Change the features to the target shape of fm_old intermediate input
            if args.fm_old in ['labram']:
                if 'ISRUC' in args.data_dir:
                    bridge_output = bridge_output.reshape(current_batch_size, 181, 200)
                elif 'FOG' in args.data_dir:
                    bridge_output = bridge_output.reshape(current_batch_size, 19, 200)
            elif args.fm_old in ['papagei']:
                pass
            elif args.fm_old in ['ecgfm']:
                bridge_output = bridge_output.permute(2, 0, 1)
            elif args.fm_old in ['hubertecg', 'hubertecgb', 'hubertecgl']:
                bridge_output = bridge_output.permute(0, 2, 1)
            elif args.fm_old in ['ecgdualnet']:
                bridge_output = bridge_output.permute(2, 0, 1)
            elif args.fm_old in ['normwear']:
                if any(loc in args.bridge_output_location for loc in ['encoder_blocks']):
                    if 'emgfmnormwear' in args.data_dir:
                        bridge_output = bridge_output.permute(0, 2, 1)
                        bridge_output = bridge_output.view(current_batch_size, 3, 560, 768)
                        bridge_output = bridge_output.reshape(current_batch_size * 3, 560, 768)
                elif 'linear_prober' in args.bridge_output_location:
                    bridge_output = bridge_output.view(current_batch_size, 768)
                        
            # Calculate bridge loss
            if args.bridge_criterion_location in ['bridge_output']:
                if args.bridge_criterion in ['CosineEmbeddingLoss']:
                    loss_bridge = bridge_criterion(bridge_output.view(current_batch_size, -1), fm_old_input.view(current_batch_size, -1), torch.ones(current_batch_size).to(device))
                elif args.bridge_criterion in ['CosineEmbeddingLoss_tokenwise']:
                    bridge_output = bridge_output.view(-1, 200)
                    fm_old_input = fm_old_input.view(-1, 200)
                    loss_bridge = bridge_criterion(bridge_output, fm_old_input, torch.ones(fm_old_input.size(0)).to(device))
                else:
                    loss_bridge = bridge_criterion(bridge_output.view(current_batch_size, -1), fm_old_input.view(current_batch_size, -1))
                messenger['bridge_output'] = bridge_output
                if 'linear_prober' not in args.bridge_output_location:
                    hook_handle = eval(f"fm_old.{args.bridge_output_location}").register_forward_pre_hook(
                        create_forward_pre_hook(messenger)
                    )
                    messenger = eval(f"{args.fm_old}_forward")(fm=fm_old, data=data_old, messenger=messenger)
                    fm_old_lastlayer_output_pred = messenger['fm_lastlayer_output']
                    hook_handle.remove()
                    # Reshape for normwear
                    if args.fm_old in ['normwear']:
                        BN, P, E = fm_old_lastlayer_output_pred.shape
                        fm_old_lastlayer_output_pred = torch.reshape(fm_old_lastlayer_output_pred,(data_old.shape[0], -1, P, E))
                elif 'linear_prober' in args.bridge_output_location:
                    fm_old_lastlayer_output_pred = bridge_output
                    if step == 0: # only calculate the dimension to reduce at the first step
                        probe_dims_to_reduce = [i for i, dim in enumerate(fm_old_lastlayer_output.shape) if dim != current_batch_size and dim != args.linear_probe_input_dim]
                    fm_old_lastlayer_output = torch.mean(fm_old_lastlayer_output, dim=tuple(probe_dims_to_reduce))
            elif args.bridge_criterion_location in ['fm_old_lastlayer_output']:
                # Send bridge_output to fm_old and get fm_old lastlayer feature
                messenger['bridge_output'] = bridge_output #fm_old_input.view(current_batch_size, 181, 200) #bridge_output #torch.zeros_like(bridge_output, device=device)
                if 'linear_prober' not in args.bridge_output_location:
                    hook_handle = eval(f"fm_old.{args.bridge_output_location}").register_forward_pre_hook(
                        create_forward_pre_hook(messenger)
                    )
                    messenger = eval(f"{args.fm_old}_forward")(fm=fm_old, data=data_old, messenger=messenger)
                    fm_old_lastlayer_output_pred = messenger['fm_lastlayer_output']
                    hook_handle.remove()
                    # Reshape for normwear
                    if args.fm_old in ['normwear']:
                        BN, P, E = fm_old_lastlayer_output_pred.shape
                        fm_old_lastlayer_output_pred = torch.reshape(fm_old_lastlayer_output_pred,(data_old.shape[0], -1, P, E))
                elif 'linear_prober' in args.bridge_output_location:
                    fm_old_lastlayer_output_pred = bridge_output
                    if step == 0: # only calculate the dimension to reduce at the first step
                        probe_dims_to_reduce = [i for i, dim in enumerate(fm_old_lastlayer_output.shape) if dim != current_batch_size and dim != args.linear_probe_input_dim]
                    fm_old_lastlayer_output = torch.mean(fm_old_lastlayer_output, dim=tuple(probe_dims_to_reduce))

                
                # Calculate the loss
                if args.bridge_criterion in ['CosineEmbeddingLoss']:
                    loss_bridge = bridge_criterion(fm_old_lastlayer_output_pred.view(current_batch_size, -1), fm_old_lastlayer_output.view(current_batch_size, -1), torch.ones(current_batch_size).to(device))
                elif args.bridge_criterion in ['CosineEmbeddingLoss_tokenwise']:
                    fm_old_lastlayer_output_pred = fm_old_lastlayer_output_pred.view(-1, 200)
                    fm_old_lastlayer_output = fm_old_lastlayer_output.view(-1, 200)
                    loss_bridge = bridge_criterion(fm_old_lastlayer_output_pred, fm_old_lastlayer_output, torch.ones(fm_old_lastlayer_output.size(0)).to(device))
                else:
                    loss_bridge = bridge_criterion(fm_old_lastlayer_output_pred.view(current_batch_size, -1), fm_old_lastlayer_output.view(current_batch_size, -1))
            
        elif args.mode in ['linear_probe_fm_new']:
            # Forward for FMs
            fm_old_input, fm_new_output, fm_old_lastlayer_output = fm_forward(fm_old=None, fm_new=fm_new, data_old=None, data_new=data_new, messenger=messenger, args=args)
        elif args.mode in ['check_before_train']:
            # Forward for FMs
            fm_old_input, fm_new_output, fm_old_lastlayer_output = fm_forward(fm_old=fm_old, fm_new=None, data_old=data_old, data_new=None, messenger=messenger, args=args)
       

        # Compute classification results from bridge features using old FM
        if args.mode in ['train', 'evaluate']:
            # Prepare the features from FM for input to the linear prober
            if args.linear_probe_input_dim_reduction in ['mean']:    
                if 'linear_prober' not in args.bridge_output_location:
                    if step == 0: # only calculate the dimension to reduce at the first step
                        probe_dims_to_reduce = [i for i, dim in enumerate(fm_old_lastlayer_output_pred.shape) if dim != current_batch_size and dim != args.linear_probe_input_dim]
                    fm_old_lastlayer_output_pred = torch.mean(fm_old_lastlayer_output_pred, dim=tuple(probe_dims_to_reduce))
                elif 'linear_prober' in args.bridge_output_location:
                    pass
            elif args.linear_probe_input_dim_reduction is None:
                fm_old_lastlayer_output_pred = fm_old_lastlayer_output_pred.view(current_batch_size, -1)
            
            # Forward with the linear prober
            output = linear_prober(fm_old_lastlayer_output_pred)
            loss_logits = logits_criterion(output, target)

        elif args.mode in ['linear_probe_fm_new']:
            # Prepare the features from FM for input to the linear prober
            if args.linear_probe_input_dim_reduction in ['mean']:
                if step == 0: # only calculate the dimension to reduce at the first step
                    probe_dims_to_reduce = [i for i, dim in enumerate(fm_new_output.shape) if dim != current_batch_size and dim != args.linear_probe_input_dim]
                fm_new_output = torch.mean(fm_new_output, dim=tuple(probe_dims_to_reduce))
            elif args.linear_probe_input_dim_reduction is None:
                fm_new_output = fm_new_output.view(current_batch_size, -1)
            
            # Forward with the linear prober
            output = linear_prober(fm_new_output)
            loss_logits = logits_criterion(output, target)
        elif args.mode in ['check_before_train']:
            # Prepare the features from FM for input to the linear prober
            if args.linear_probe_input_dim_reduction in ['mean']:
                if step == 0: # only calculate the dimension to reduce at the first step
                    probe_dims_to_reduce = [i for i, dim in enumerate(fm_old_lastlayer_output.shape) if dim != current_batch_size and dim != args.linear_probe_input_dim]
                fm_old_lastlayer_output = torch.mean(fm_old_lastlayer_output, dim=tuple(probe_dims_to_reduce))
            elif args.linear_probe_input_dim_reduction is None:
                fm_old_lastlayer_output = fm_old_lastlayer_output.view(current_batch_size, -1)
            
            # Forward with the linear prober
            output = linear_prober(fm_old_lastlayer_output)
            loss_logits = logits_criterion(output, target)


        if is_binary:
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()
        target = target.cpu()

        pred.append(output)
        true.append(target)
        
        if args.mode in ['train', 'evaluate']:
            metric_logger.update(loss_logits=loss_logits.item())
            metric_logger.update(loss_bridge=loss_bridge.item())
        elif args.mode in ['linear_probe_fm_new', 'check_before_train']:
            metric_logger.update(loss_logits=loss_logits.item())
        
        # Optional testrun debug
        if args.debug_run and step == 2:
            break

    
    # Calculate scores for the desired metrics
    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()
    ret = get_metrics(pred, true, metrics, is_binary)
    
    if args.mode in ['train', 'evaluate']:
        ret['loss_logits'] = metric_logger.loss_logits.global_avg
        ret['loss_bridge'] = metric_logger.loss_bridge.global_avg
        cf_matrix = confusion_matrix(true, np.argmax(pred,axis=1))
        class_names = sorted(set(true).union(set(np.argmax(pred,axis=1))))
        cm_percent = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=class_names,columns=class_names)
        cm_abs = pd.DataFrame(cf_matrix, index=class_names,columns=class_names)
        ret['cm_percent'] = cm_percent
        ret['cm_abs'] = cm_abs
    elif args.mode in ['linear_probe_fm_new', 'check_before_train']:
        ret['loss_logits'] = metric_logger.loss_logits.global_avg
        cf_matrix = confusion_matrix(true, np.argmax(pred,axis=1))
        class_names = sorted(set(true).union(set(np.argmax(pred,axis=1))))
        cm_percent = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=class_names,columns=class_names)
        cm_abs = pd.DataFrame(cf_matrix, index=class_names,columns=class_names)
        ret['cm_percent'] = cm_percent
        ret['cm_abs'] = cm_abs
    return ret


def create_forward_pre_hook(messenger):
    def replace_input_hook(module, input):
        # create a copy of input and replace the first element with the new input
        new_input = list(input)
        new_input[0] = messenger['bridge_output']
        return tuple(new_input)  # Replace the input to layer
    return replace_input_hook

def fm_input_hook(module, input, output, messenger):
    messenger['fm_input'] = input

def fm_output_hook(module, input, output, messenger):
    if isinstance(output, tuple):
        messenger['fm_output'] = output[0]
    else:
        messenger['fm_output'] = output

def fm_lastlayer_output_hook(module, input, output, messenger):
    if isinstance(output, tuple):
        messenger['fm_lastlayer_output'] = output[0]
    else:
        messenger['fm_lastlayer_output'] = output

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots a confusion matrix using seaborn and matplotlib.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt.gcf()