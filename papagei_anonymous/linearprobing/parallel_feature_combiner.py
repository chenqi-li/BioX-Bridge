# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import subprocess
import argparse

def run_combiner(model_path, split, level, content):
    result = subprocess.run(
        ['python3', 'extracted_feature_combine.py', model_path, split, level, content],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('level', type=str, help="patient or segment")
    parser.add_argument('content', type=str)
    parser.add_argument(
        '-d', '--datasets',
        nargs='+',  # '+' means one or more values
        type=str,  # assuming the list will be of strings
        help="List of dataset strings"
    )
    args = parser.parse_args()
    splits = ['train', 'val', 'test']
    print(args.datasets)
    for dataset in args.datasets:
        for split in splits:
            embedding_path = f"../../data/{dataset}/features/{args.model_name}"
            run_combiner(embedding_path, split, args.level, args.content) 
