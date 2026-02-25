# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np 
import pandas as pd 
import joblib 
import os
import torch
import sys
from tqdm import tqdm
from momentfm import MOMENTPipeline
sys.path.append("../")
from utilities import get_data_info
import argparse

def none_or_int(value):
    if value == 'None':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: '{value}'")

def batch_load_signals(path, case, segments):
    batch_signal = []
    for s in segments:
        batch_signal.append(joblib.load(os.path.join(path, case, str(s))))
    return np.vstack(batch_signal)

def compute_signal_embeddings(model, path, case, batch_size, device, average=True):
    segments = os.listdir(os.path.join(path, case))
    embeddings = []
    model.eval()
    
    with torch.inference_mode():
        for i in range(0, len(segments), batch_size):
            batch_signal = batch_load_signals(path, case, segments[i:i+batch_size])
            batch_signal = torch.Tensor(batch_signal).unsqueeze(dim=1).to(device)
            embeddings.append(model(batch_signal).embeddings.cpu().detach().numpy())

    embeddings = np.vstack(embeddings)
    return embeddings

def get_embeddings(path, child_dirs, save_dir, model, batch_size, device, average=True):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f"[INFO] Creating directory: {save_dir}")
    else:
        print(f"[INFO] {save_dir} already exists")

    dict_embeddings = {}
    for i in tqdm(range(len(child_dirs))):
        case = str(child_dirs[i])
        embeddings = compute_signal_embeddings(model=model,
                                              path=path,
                                              case=case,
                                              batch_size=batch_size,
                                              device=device,
                                              average=average)
        print(f"[INFO] Saving file {case} to {save_dir}")
        joblib.dump(embeddings, os.path.join(save_dir, case + ".p"))

def compute_signal_embeddings_df(model, path, case, segments, batch_size, device, average=True):
    embeddings = []
    model.eval()
    
    with torch.inference_mode():
        for i in range(0, len(segments), batch_size):
            batch_signal = batch_load_signals(path, case, segments[i:i+batch_size])
            batch_signal = torch.Tensor(batch_signal).unsqueeze(dim=1).to(device)
            embeddings.append(model(batch_signal).embeddings.cpu().detach().numpy())

    embeddings = np.vstack(embeddings)
    return embeddings


def get_embeddings_df(path, df, case_name, child_dirs, save_dir, model, batch_size, device, average=True):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f"[INFO] Creating directory: {save_dir}")
    else:
        print(f"[INFO] {save_dir} already exists")

    dict_embeddings = {}
    for i in tqdm(range(len(child_dirs))):
        segments = df[df[case_name] == child_dirs[i]].segments.values
        case = str(child_dirs[i])
        embeddings = compute_signal_embeddings_df(model=model,
                                              path=path,
                                              case=case,
                                              segments=segments,
                                              batch_size=batch_size,
                                              device=device,
                                              average=average)
        print(f"[INFO] Saving file {case} to {save_dir}")
        joblib.dump(embeddings, os.path.join(save_dir, case + ".p"))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('device', type=str, help="CUDA device for model")
    parser.add_argument('dataset', type=str, help="Dataset to extract")
    parser.add_argument('split', type=str, help="Data split to process")
    parser.add_argument('save_dir', type=str, help="Path to the save directory")
    parser.add_argument('start_idx', type=none_or_int, default=None)
    parser.add_argument('end_idx', type=none_or_int, default=None)
    args = parser.parse_args()

    if args.dataset in ["vital", "mimic", "mesa", "wesad", "dalia"]:
        df_train, df_val, df_test, case_name, ppg_dir = get_data_info(args.dataset, prefix="../", usecolumns=['segments'])
    else:
        df_train, df_val, df_test, case_name, ppg_dir = get_data_info(args.dataset, prefix="../")    
    
    dict_df = {'train': df_train, 'val': df_val, 'test': df_test}
    df = dict_df[args.split]
    child_dirs = np.unique(df[case_name].values)[args.start_idx:args.end_idx]
    save_dir = f"{args.save_dir}/moment/{args.split}/"

    if not os.path.exists(f"{args.save_dir}/moment"):
        os.mkdir(f"{args.save_dir}/moment")

    model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={"task_name": "embedding"},
    )
    model.init()
    device = f"cuda:{args.device}"
    model.to(device)
    batch_size = 300

    if args.dataset in ["vital", "mimic", "mesa", "wesad", "dalia", "bidmc"]:
        get_embeddings_df(path=ppg_dir,
            df=df,
            case_name=case_name,
            child_dirs=child_dirs,
            save_dir=save_dir,
            model=model,
            batch_size=batch_size,
            device=device)
    else:
        get_embeddings(path=ppg_dir,
                        child_dirs=child_dirs,
                        save_dir=save_dir,
                        model=model,
                        batch_size=batch_size,
                        device=device)
