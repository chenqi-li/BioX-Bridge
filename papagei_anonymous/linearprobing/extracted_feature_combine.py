# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
Combine the the extracted segment-level embeddings to participant level
"""

import joblib 
import numpy as np
import pandas as pd
import os 
import argparse
from tqdm import tqdm 

def segment_avg_to_dict(path, level):
    
    filenames = os.listdir(path)
    dict_data = {}
    
    for i in tqdm(range(len(filenames))):
        f = filenames[i]
        data = joblib.load(os.path.join(path, f))
        if level == "patient":
            dict_data[f.split(".")[0]] = np.mean(data, axis=0)
        else:
            dict_data[f.split(".")[0]] = data

    return dict_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('main_dir', type=str, help="Dataset to extract")
    parser.add_argument('split', type=str, help="Data split to process")
    parser.add_argument('level', type=str, help="patient or segment")
    parser.add_argument('content', type=str, help="extra notes")
    args = parser.parse_args()

    path = f"{args.main_dir}/{args.split}/"
    dict_feat = segment_avg_to_dict(path, args.level)
    joblib.dump(dict_feat, f"{args.main_dir}/dict_{args.split}{args.content}.p")