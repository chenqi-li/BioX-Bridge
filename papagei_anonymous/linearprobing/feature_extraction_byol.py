# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np 
import pandas as pd 
import joblib 
import os
import torch
import sys
import torch
import argparse
import torch.fft as fft
sys.path.append("../models")
sys.path.append("../")
from utilities import get_data_info
from utils import load_model_without_module_prefix, batch_load_signals, resample_batch_signal, none_or_int, str2bool
from tqdm import tqdm
from resnet import ResNet1D, ResNet1DMoE, TFCResNet
from transformer import TransformerSimple
from augmentations import ResampleSignal
from torch_ecg._preprocessors import Normalize
from baselines.BYOL.byol_architecture import BYOL

def extract_byol_features(model, input_data, device):
    
    input_data = input_data.to(device)
    net = model.online_encoder.net
    
    x = net.first_block_conv(input_data)
    x = net.first_block_bn(x)
    x = net.first_block_relu(x)
    
    for block in net.basicblock_list:
        x = block(x)
    
    x = net.final_bn(x)
    x = net.final_relu(x)
    x = x.mean(-1)
    features = net.dense(x)

    return features

def compute_signal_embeddings(model, path, case, segments, batch_size, device, resample=False, normalize=True, fs=None, fs_target=None):
    embeddings = []
    model.eval()
    norm = Normalize(method='z-score')

    with torch.inference_mode():
        for i in range(0, len(segments), batch_size):
            batch_signal = batch_load_signals(path, case, segments[i:i+batch_size])
            if normalize:
                batch_signal = np.vstack([norm.apply(s, fs)[0] for s in batch_signal])
            if resample:
                batch_signal = resample_batch_signal(batch_signal, fs, fs_target)
            batch_signal = torch.Tensor(batch_signal).unsqueeze(dim=1).to(device)
            features = extract_byol_features(model, batch_signal, device).cpu().detach().numpy()
            embeddings.append(features)

    embeddings = np.vstack(embeddings)

    return embeddings

def save_embeddings(path, df, child_dirs, save_dir, model, batch_size, device, is_directory, resample=False, normalize=True, fs=None, fs_target=None):
    dict_embeddings = {}

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f"[INFO] Creating directory: {save_dir}")
    else:
        print(f"[INFO] {save_dir} already exists")

    for i in tqdm(range(len(child_dirs))):
        case = str(child_dirs[i])
        if is_directory:
            segments = os.listdir(os.path.join(path, case))
        else:
            print("Extracting segments from df")
            segments = df[df[case_name] == child_dirs[i]].segments.values

        embeddings = compute_signal_embeddings(model=model,
                                            path=path,
                                            case=case,
                                            segments=segments,
                                            batch_size=batch_size,
                                            device=device,
                                            resample=resample,
                                            normalize=normalize,
                                            fs=fs,
                                            fs_target=fs_target,
                                            )
                                    
        print(f"[INFO] Saving file {case} to {save_dir}")
        joblib.dump(embeddings, os.path.join(save_dir, case + ".p"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('device', type=str, help="CUDA device for model")
    parser.add_argument('dataset', type=str, help="Dataset to extract")
    parser.add_argument('split', type=str, help="Data split to process")
    parser.add_argument('start_idx', type=none_or_int, default=None)
    parser.add_argument('end_idx', type=none_or_int, default=None)
    parser.add_argument('resample', type=str2bool, default=None)
    parser.add_argument('normalize', type=str2bool, default=None)
    parser.add_argument('fs', type=float, default=None)
    parser.add_argument('fs_target', type=int, default=None)
    args = parser.parse_args()

    print(f"Resample: {args.resample} | Normalize: {args.normalize}")

    batch_size = 256
    model_config = {'base_filters': 32,
                'kernel_size': 3,
                'stride': 2,
                'groups': 1,
                'n_block': 18,
                'n_classes': 512,
                }

    net = ResNet1D(in_channels=1, 
                base_filters=model_config['base_filters'], 
                kernel_size=model_config['kernel_size'],
                stride=model_config['stride'],
                groups=model_config['groups'],
                n_block=model_config['n_block'],
                n_classes=model_config['n_classes'])

    model = BYOL(image_size=1250,
            net=net)
    model = load_model_without_module_prefix(model, "../../models/2024_09_16_14_19_52/resnet_byol_mkhl43_2024_09_16_14_19_52_step10000_loss1.2126.pt")
    device = f"cuda:{args.device}"
    model.to(device)

    if args.dataset in ["vital", "mimic", "mesa", "wesad", "dalia"]:
        df_train, df_val, df_test, case_name, ppg_dir = get_data_info(args.dataset, prefix="../", usecolumns=['segments'])
    else:
        df_train, df_val, df_test, case_name, ppg_dir = get_data_info(args.dataset, prefix="../")

    dict_df = {'train': df_train, 'val': df_val, 'test': df_test}
    df = dict_df[args.split]
    child_dirs = np.unique(df[case_name].values)[args.start_idx:args.end_idx]

    save_dir = f"../../data/{args.dataset}/features/byol"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = f"{save_dir}/{args.split}/"

    is_directory=True
    if args.dataset in ["vital", "mimic", "mesa", "wesad", "dalia", "marsh"]:
        is_directory=False

    save_embeddings(path=ppg_dir,
               df=df,
               child_dirs=child_dirs,
               save_dir=save_dir,
               model=model,
               batch_size=batch_size,
               device=device,
               is_directory=is_directory,
               resample=args.resample,
               normalize=args.normalize,
               fs=args.fs,
               fs_target=args.fs_target)
