# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import subprocess
import torch
from concurrent.futures import ProcessPoolExecutor

MODEL_PATH = "../../models/2024_09_24_01_05_17/resnet_mt_moe_r36_<dataset.PPGDatasetLabelsArray object at 0x7fda63432470>__kwdjiu_2024_09_24_01_05_17_step9837_loss0.1493.pt"
ARCHITECUTRE = "resnet_moe"
is_mt_regress = "False"
DATASETS = ["vital", "mesa", "mimic", "sdb", "ppg-bp", "wesad", "dalia", "ecsmp", "numom2b", "vv"]
SPLITS = ["train", "test", "val"]
fs_target = "125"
output_idx = "0" # for papagei

def get_argument_combinations(dataset, split):
    if dataset == "vital":  
        resample = "True"
        fs = "500"
        normalize = "True"

    if dataset == "mimic":
        resample = "False"
        fs = "125"
        normalize = "True"

    if dataset == "vital" or dataset == "mimic":
        if split == "val" or split == "test":
            return [
                (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "100", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "100", "200", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "200", "300", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "4", dataset, split, f"/workspace/data/{dataset}/features", "300", "400", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "5", dataset, split, f"/workspace/data/{dataset}/features", "400", "500", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "6", dataset, split, f"/workspace/data/{dataset}/features", "500", "60000", resample, normalize, fs, fs_target),
            ]
        else:
            return [
                (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "750", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "750", "1500", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "1500", "2250", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "4", dataset, split, f"/workspace/data/{dataset}/features", "2250", "3000", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "5", dataset, split, f"/workspace/data/{dataset}/features", "3000", "3750", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "6", dataset, split, f"/workspace/data/{dataset}/features", "3750", "4500", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "7", dataset, split, f"/workspace/data/{dataset}/features", "4500", "45000", resample, normalize, fs, fs_target),
            ]

    if dataset == "mesa":
        resample = "True"
        fs = "256"
        normalize="True"
        if split == "test" or split == "val":
            return [
                (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "25", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "25", "50", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "50", "75", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "4", dataset, split, f"/workspace/data/{dataset}/features", "75", "100", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "5", dataset, split, f"/workspace/data/{dataset}/features", "100", "125", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "6", dataset, split, f"/workspace/data/{dataset}/features", "125", "150", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "7", dataset, split, f"/workspace/data/{dataset}/features", "150", "175", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "0", dataset, split, f"/workspace/data/{dataset}/features", "175", "20000", resample, normalize, fs, fs_target),
            ]
        else:
            return [
                (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "300", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "300", "600", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "600", "900", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "4", dataset, split, f"/workspace/data/{dataset}/features", "900", "1200", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "5", dataset, split, f"/workspace/data/{dataset}/features", "1200", "1500", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "6", dataset, split, f"/workspace/data/{dataset}/features", "1500", "1800", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "7", dataset, split, f"/workspace/data/{dataset}/features", "1800", "20000", resample, normalize, fs, fs_target),
            ]

    if dataset == "sdb":
        resample = "True"
        fs = "62.5"
        normalize = "True"
        if split == "train":
            return [
                    (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "20", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "20", "40", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "40", "60", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "0", dataset, split, f"/workspace/data/{dataset}/features", "60", "20000", resample, normalize, fs, fs_target),
                ]
        else:
            return [
                    (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "10", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "10", "20", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "20", "30", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "0", dataset, split, f"/workspace/data/{dataset}/features", "30", "20000", resample, normalize, fs, fs_target),
                ]

    if dataset == "ppg-bp":
        resample = "False"
        fs = "125"
        normalize = "False"
        if split == "train":
            return [
                    (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "20", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "20", "40", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "40", "60", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "0", dataset, split, f"/workspace/data/{dataset}/features", "60", "20000", resample, normalize, fs, fs_target),
                ]
        else:
            return [
                    (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "10", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "10", "20", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "20", "30", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "0", dataset, split, f"/workspace/data/{dataset}/features", "30", "20000", resample, normalize, fs, fs_target),
                ]
    
    if dataset == "ecsmp":
        resample = "False"
        fs = "64"
        normalize = "False"
        if split == "val" or split == "test":
            return [
                    (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "4", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "4", "8", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "8", "12", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "0", dataset, split, f"/workspace/data/{dataset}/features", "12", "20000", resample, normalize, fs, fs_target),
                ]
        else:
            return [
                    (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "7", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "7", "14", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "14", "21", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "0", dataset, split, f"/workspace/data/{dataset}/features", "21", "28", resample, normalize, fs, fs_target),
                    (ARCHITECUTRE, MODEL_PATH, "4", dataset, split, f"/workspace/data/{dataset}/features", "28", "35", resample, normalize, fs, fs_target),
                ]

    if dataset == "wesad" or dataset == "dalia":
        resample = "False"
        fs = "64"
        normalize="False"
        if split == "test" or split == "val":
            return [
                (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "1", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "1", "2", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "2", "75", resample, normalize, fs, fs_target),
            ]
        else:
            return [
                (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "2", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "2", "4", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "4", "6", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "4", dataset, split, f"/workspace/data/{dataset}/features", "6", "8", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "5", dataset, split, f"/workspace/data/{dataset}/features", "8", "1500", resample, normalize, fs, fs_target),
            ]

    if dataset == "numom2b":
        resample = "False"
        fs = "200"
        normalize="False"
        if split == "test" or split == "val":
            return [
                (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "150", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "150", "300", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "300", "450", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "4", dataset, split, f"/workspace/data/{dataset}/features", "450", "600", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "5", dataset, split, f"/workspace/data/{dataset}/features", "600", "750", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "6", dataset, split, f"/workspace/data/{dataset}/features", "750", "900", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "7", dataset, split, f"/workspace/data/{dataset}/features", "900", "45000", resample, normalize, fs, fs_target),
            ]
        else:
            return [
                (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "450", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "450", "900", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "900", "1350", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "4", dataset, split, f"/workspace/data/{dataset}/features", "1350", "1800", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "5", dataset, split, f"/workspace/data/{dataset}/features", "1800", "2250", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "6", dataset, split, f"/workspace/data/{dataset}/features", "2250", "2700", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "7", dataset, split, f"/workspace/data/{dataset}/features", "2700", "40000", resample, normalize, fs, fs_target),
            ]

    if dataset == "bidmc" or dataset == "mimicAF":
        resample = "False"
        fs = "125"
        normalize="False"
        if split == "test" or split == "val":
            return [
                (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "3", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "3", "6", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "6", "450", resample, normalize, fs, fs_target),
            ]
        else:
            return [
                (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "5", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "5", "10", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "10", "15", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "4", dataset, split, f"/workspace/data/{dataset}/features", "15", "20", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "5", dataset, split, f"/workspace/data/{dataset}/features", "20", "2250", resample, normalize, fs, fs_target),
            ]

    if dataset == "vv":
        resample = "False"
        fs = "60"
        normalize="False"
        if split == "test" or split == "val":
            return [
                (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "7", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "7", "14", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "14", "21", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "4", dataset, split, f"/workspace/data/{dataset}/features", "21", "28", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "5", dataset, split, f"/workspace/data/{dataset}/features", "28", "35", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "6", dataset, split, f"/workspace/data/{dataset}/features", "35", "42", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "7", dataset, split, f"/workspace/data/{dataset}/features", "42", "450", resample, normalize, fs, fs_target),
            ]
        else:
            return [
                (ARCHITECUTRE, MODEL_PATH, "1", dataset, split, f"/workspace/data/{dataset}/features", "0", "20", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"/workspace/data/{dataset}/features", "20", "40", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "3", dataset, split, f"/workspace/data/{dataset}/features", "40", "60", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "4", dataset, split, f"/workspace/data/{dataset}/features", "60", "80", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "5", dataset, split, f"/workspace/data/{dataset}/features", "80", "100", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "6", dataset, split, f"/workspace/data/{dataset}/features", "100", "120", resample, normalize, fs, fs_target),
                (ARCHITECUTRE, MODEL_PATH, "7", dataset, split, f"/workspace/data/{dataset}/features", "120", "1400", resample, normalize, fs, fs_target),
            ]


def run_papagei(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    result = subprocess.run(
        ['python', 'feature_extraction_papagei.py', architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target, is_mt_regress, output_idx],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr
    
def run_tfc(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    result = subprocess.run(
        ['python', 'feature_extraction_tfc.py', device, dataset, split, start_idx, end_idx, resample, normalize, fs, fs_target],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr

def run_byol(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    result = subprocess.run(
        ['python', 'feature_extraction_byol.py', device, dataset, split, start_idx, end_idx, resample, normalize, fs, fs_target],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr

def run_moment(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    # The arguments are given but not used for ease of use.
    # E.g., model_path is given, but it does not matter.
    result = subprocess.run(
        ['python', 'feature_extraction_moment.py', device, dataset, split, save_dir, start_idx, end_idx],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr

def run_chronos(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    # The arguments are given but not used for ease of use.
    # E.g., model_path is given, but it does not matter.
    result = subprocess.run(
        ['python', 'feature_extraction_chronos.py', device, dataset, split, save_dir, start_idx, end_idx],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr

if __name__ == "__main__":
    torch.cuda.empty_cache() 
    with ProcessPoolExecutor(max_workers=8) as executor:  # Adjust max_workers based on your GPU capacity
        futures = []
        for dataset in DATASETS:
            for split in SPLITS:
                argument_combinations = get_argument_combinations(dataset, split)
                for args in argument_combinations:
                    futures.append(executor.submit(run_papagei, *args))
        
        for future in futures:
            stdout, stderr = future.result()
            print("Output:", stdout)
            if stderr:
                print("Error:", stderr)
