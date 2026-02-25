# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np 
import pandas as pd 
import joblib 
import os
import torch
import sys
import argparse
from tqdm import tqdm
from chronos import ChronosPipeline
sys.path.append("../")
from utilities import get_data_info

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

def compute_signal_embeddings_chronos(pipeline, path, case, batch_size):
    segments = os.listdir(os.path.join(path, case))
    embeddings = []
    pipeline.model.eval()
    
    with torch.inference_mode():
        for i in range(0, len(segments), batch_size):
            batch_signal = batch_load_signals(path, case, segments[i:i+batch_size])
            batch_signal = torch.tensor(batch_signal)
            embedding, _ = pipeline.embed(batch_signal)
            embeddings.append(embedding.cpu().detach().float().numpy())

    embeddings = np.mean(np.vstack(embeddings), axis=1)
    return embeddings


def get_embeddings_chronos(path, child_dirs, save_dir, pipeline, batch_size):
    dict_embeddings = {}

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f"[INFO] Creating directory: {save_dir}")
    else:
        print(f"[INFO] {save_dir} already exists")

    for i in tqdm(range(len(child_dirs))):
        case = str(child_dirs[i])
        embeddings = compute_signal_embeddings_chronos(pipeline=pipeline,
                                                  path=path,
                                                  case=case,
                                                  batch_size=batch_size)
        
        print(f"[INFO] Saving file {case} to {save_dir}")
        joblib.dump(embeddings, os.path.join(save_dir, case + ".p"))


def compute_signal_embeddings_chronos_df(pipeline, path, case, segments, batch_size):
    embeddings = []
    pipeline.model.eval()
    
    with torch.inference_mode():
        for i in range(0, len(segments), batch_size):
            batch_signal = batch_load_signals(path, case, segments[i:i+batch_size])
            batch_signal = torch.tensor(batch_signal)
            embedding, _ = pipeline.embed(batch_signal)
            embeddings.append(embedding.cpu().detach().float().numpy())

    embeddings = np.mean(np.vstack(embeddings), axis=1)
    return embeddings

def get_embeddings_chronos_df(path, df, case_name, child_dirs, save_dir, pipeline, batch_size):
    dict_embeddings = {}

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f"[INFO] Creating directory: {save_dir}")
    else:
        print(f"[INFO] {save_dir} already exists")

    for i in tqdm(range(len(child_dirs))):
        segments = df[df[case_name] == child_dirs[i]].segments.values
        case = str(child_dirs[i])
        embeddings = compute_signal_embeddings_chronos_df(pipeline=pipeline,
                                                        path=path,
                                                        case=case,
                                                        segments=segments,
                                                        batch_size=batch_size)
        
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
    save_dir = f"{args.save_dir}/chronos/{args.split}/"
    batch_size = 128

    if not os.path.exists(f"{args.save_dir}/chronos"):
        os.mkdir(f"{args.save_dir}/chronos")

    pipeline = ChronosPipeline.from_pretrained(
                            "amazon/chronos-t5-base",
                            device_map=f"cuda:{args.device}",
                            torch_dtype=torch.bfloat16)
    
    if args.dataset in ["vital", "mimic", "mesa", "wesad", "dalia", "bidmc"]:
        get_embeddings_chronos_df(path=ppg_dir,
                        df=df,
                        case_name=case_name,
                        child_dirs=child_dirs,
                        save_dir=save_dir,
                        pipeline=pipeline,
                        batch_size=batch_size)
    else:
        get_embeddings_chronos(path=ppg_dir,
                        child_dirs=child_dirs,
                        save_dir=save_dir,
                        pipeline=pipeline,
                        batch_size=batch_size)