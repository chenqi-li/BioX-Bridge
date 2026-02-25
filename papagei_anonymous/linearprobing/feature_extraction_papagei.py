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
sys.path.append("../models")
sys.path.append("../")
from utilities import get_data_info
from .utils import load_model_without_module_prefix, batch_load_signals, resample_batch_signal, none_or_int, str2bool
from tqdm import tqdm
from models.resnet import ResNet1D, ResNet1DMoE
from models.transformer import TransformerSimple
from augmentations import ResampleSignal
from torch_ecg._preprocessors import Normalize

def compute_signal_embeddings(model, path, case, batch_size, device, output_idx, resample=False, normalize=True, fs=None, fs_target=None, is_mt_regress=False):
    segments = os.listdir(os.path.join(path, case))
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
            outputs = model(batch_signal)
            embeddings.append(outputs[output_idx].cpu().detach().numpy())

    embeddings = np.vstack(embeddings)

    return embeddings

def save_embeddings(path, child_dirs, save_dir, model, batch_size, device, output_idx, resample=False, normalize=True, fs=None, fs_target=None, is_mt_regress=False):
    dict_embeddings = {}

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f"[INFO] Creating directory: {save_dir}")
    else:
        print(f"[INFO] {save_dir} already exists")

    for i in tqdm(range(len(child_dirs))):
        case = str(child_dirs[i])
        embeddings = compute_signal_embeddings(model=model,
                                            path=path,
                                            case=case,
                                            batch_size=batch_size,
                                            device=device,
                                            output_idx=output_idx,
                                            resample=resample,
                                            normalize=normalize,
                                            fs=fs,
                                            fs_target=fs_target,
                                            is_mt_regress=is_mt_regress,
                                            )
                                    
        print(f"[INFO] Saving file {case} to {save_dir}")
        joblib.dump(embeddings, os.path.join(save_dir, case + ".p"))

def compute_signal_embeddings_df(model, path, case, segments, batch_size, device, output_idx, resample=False, normalize=True, fs=None, fs_target=None, is_mt_regress=False):
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
            outputs = model(batch_signal)
            embeddings.append(outputs[output_idx].cpu().detach().numpy())


    embeddings = np.vstack(embeddings)

    return embeddings

def save_embeddings_df(path, df, case_name, child_dirs, save_dir, model, batch_size, device, output_idx, resample=False, normalize=True, fs=None, fs_target=None, is_mt_regress=False):
    dict_embeddings = {}

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f"[INFO] Creating directory: {save_dir}")
    else:
        print(f"[INFO] {save_dir} already exists")

    for i in tqdm(range(len(child_dirs))):
        segments = df[df[case_name] == child_dirs[i]].segments.values
        case = str(child_dirs[i])
        embeddings = compute_signal_embeddings_df(model=model,
                                            path=path,
                                            case=case,
                                            segments=segments,
                                            batch_size=batch_size,
                                            device=device,
                                            resample=resample,
                                            normalize=normalize,
                                            fs=fs,
                                            fs_target=fs_target,
                                            is_mt_regress=is_mt_regress,
                                            output_idx=output_idx
                                            )
        
        print(f"[INFO] Saving file {case} to {save_dir}")
        joblib.dump(embeddings, os.path.join(save_dir, case + ".p"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('architecture', type=str, help="resnet or transformer")
    parser.add_argument('model_path', type=str, help="Path to the model")
    parser.add_argument('device', type=str, help="CUDA device for model")
    parser.add_argument('dataset', type=str, help="Dataset to extract")
    parser.add_argument('split', type=str, help="Data split to process")
    parser.add_argument('save_dir', type=str, help="Path to the save directory")
    parser.add_argument('start_idx', type=none_or_int, default=None)
    parser.add_argument('end_idx', type=none_or_int, default=None)
    parser.add_argument('resample', type=str2bool, default=None)
    parser.add_argument('normalize', type=str2bool, default=None)
    parser.add_argument('fs', type=float, default=None)
    parser.add_argument('fs_target', type=int, default=None)
    parser.add_argument('is_mt_regress', type=str2bool, default=None)
    parser.add_argument('output_idx', type=int)
    args = parser.parse_args()

    print(f"Resample: {args.resample} | Normalize: {args.normalize}")

    if args.architecture == "transformer":
        model_config = {'d_model': 1250,
            'nhead': 2,
            'dim_feedforward': 2048,
            'trans_dropout': 0.0,
            'proj_dropout': 0.0,
            'num_layers': 2,
            'h1': 1024,
            'embedding_size': 512}
        model = TransformerSimple(model_config=model_config)
    
    if args.architecture == "resnet":
        model_config = {'base_filters': 32,
                'kernel_size': 3,
                'stride': 2,
                'groups': 1,
                'n_block': 18,
                'n_classes': 512,
                }

        model = ResNet1D(in_channels=1, 
                    base_filters=model_config['base_filters'], 
                    kernel_size=model_config['kernel_size'],
                    stride=model_config['stride'],
                    groups=model_config['groups'],
                    n_block=model_config['n_block'],
                    n_classes=model_config['n_classes'],
                    use_mt_regression=False,
                    use_projection=True)
    
    if args.architecture == "resnet_mt_regression":
        model_config = {'base_filters': 32,
                'kernel_size': 3,
                'stride': 2,
                'groups': 1,
                'n_block': 18,
                'n_classes': 512,
                }

        model = ResNet1D(in_channels=1, 
                    base_filters=model_config['base_filters'], 
                    kernel_size=model_config['kernel_size'],
                    stride=model_config['stride'],
                    groups=model_config['groups'],
                    n_block=model_config['n_block'],
                    n_classes=model_config['n_classes'])

    if args.architecture == "resnet_moe":
        # model_config = {'base_filters': 64,
        #             'kernel_size': 3,
        #             'stride': 2,
        #             'groups': 1,
        #             'n_block': 24,
        #             'n_classes': 512,
        #             'n_experts': 3
        #             }

        model_config = {'base_filters': 32,
                    'kernel_size': 3,
                    'stride': 2,
                    'groups': 1,
                    'n_block': 18,
                    'n_classes': 512,
                    'n_experts': 3
                    }

        model = ResNet1DMoE(in_channels=1, 
                    base_filters=model_config['base_filters'], 
                    kernel_size=model_config['kernel_size'],
                    stride=model_config['stride'],
                    groups=model_config['groups'],
                    n_block=model_config['n_block'],
                    n_classes=model_config['n_classes'],
                    n_experts=model_config['n_experts'])

    model = load_model_without_module_prefix(model, args.model_path)
    device = f"cuda:{args.device}"
    model.to(device)

    if args.dataset in ["vital", "mimic", "mesa", "wesad", "dalia"]:
        df_train, df_val, df_test, case_name, ppg_dir = get_data_info(args.dataset, prefix="../", usecolumns=['segments'])
    else:
        df_train, df_val, df_test, case_name, ppg_dir = get_data_info(args.dataset, prefix="../")


    dict_df = {'train': df_train, 'val': df_val, 'test': df_test}
    df = dict_df[args.split]
    child_dirs = np.unique(df[case_name].values)[args.start_idx:args.end_idx]

    model_name = args.model_path.split("/")[-1].split(".pt")[0]
    if not os.path.exists(f"{args.save_dir}/{model_name}"):
        os.mkdir(f"{args.save_dir}/{model_name}")
    save_dir = f"{args.save_dir}/{model_name}/{args.split}/"
    batch_size = 256

    if args.dataset in ["vital", "mimic", "mesa", "wesad", "dalia", "marsh"]:
        save_embeddings_df(path=ppg_dir,
                df=df,
                case_name=case_name,
                child_dirs=child_dirs,
                save_dir=save_dir,
                model=model,
                batch_size=batch_size,
                device=device,
                output_idx=args.output_idx,
                resample=args.resample,
                normalize=args.normalize,
                fs=args.fs,
                fs_target=args.fs_target,
                is_mt_regress=args.is_mt_regress)
    else:
        save_embeddings(path=ppg_dir,
                        child_dirs=child_dirs, 
                        save_dir=save_dir, 
                        model=model, 
                        batch_size=batch_size, 
                        device=device, 
                        output_idx=args.output_idx,
                        resample=args.resample, 
                        normalize=args.normalize, 
                        fs=args.fs, 
                        fs_target=args.fs_target,
                        is_mt_regress=args.is_mt_regress)


                      