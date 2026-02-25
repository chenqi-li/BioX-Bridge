# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pandas as pd 
import numpy as np
import sys 
import json
import os
import argparse
sys.path.append("../")
from classification import classification_model
from regression import regression_model
from utils import load_linear_probe_dataset_objs, bootstrap_metric_confidence_interval, get_data_for_ml, get_data_for_ml_from_df
from utilities import get_data_info
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from time import time

def full_regression(dataset_name, model_name, label, func, linear_model, content, concat, level="patient", string_convert=True, percent=None):

    if concat:
        X_train, y_train, X_test, y_test, _, _, test_keys = load_linear_probe_dataset_objs(dataset_name=dataset_name,
                                                                         model_name=model_name,
                                                                         label=label,
                                                                         func=func,
                                                                         level=level,
                                                                         string_convert=string_convert,
                                                                         content=content)   
    else:
        X_train, y_train, X_val, y_val, X_test, y_test, _, _, test_keys = load_linear_probe_dataset_objs(dataset_name=dataset_name,
                                                             model_name=model_name,
                                                             label=label,
                                                             func=func,
                                                             level=level,
                                                             concat=False,
                                                             string_convert=string_convert,
                                                             content=content)
        X_test = np.concatenate((X_test, X_val))
        y_test = np.concatenate((y_test, y_val))
    
    if percent is not None:
        size = len(X_train)
        idx = np.random.choice(np.arange(size), size=int(percent * size))
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"Using {len(X_train)} of {size}")

    if linear_model == "ridge":
        estimator = Ridge()
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0],  # Regularization strength
            'solver': ['auto', 'cholesky', 'sparse_cg']  # Solver to use in the computational routines
        }

    if linear_model == "rf":
        estimator = RandomForestRegressor()
        param_grid = {
            'n_estimators': [100, 200],    
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30],     
            'min_samples_split': [2, 5],     
            'min_samples_leaf': [1, 2],      
        }

    results = regression_model(estimator=estimator,
                    param_grid=param_grid,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test)
    
    results['model'] = model_name
    results['dataset'] = dataset_name
    results['label'] = label
        
    return results

def get_results(model, config):
    all_results = []
    func = get_data_for_ml
    for key in config.keys():
        configuration = config[key]
        print(f"{configuration['dataset']} | {configuration['label']}")
        print("######################")

        results = full_regression(dataset_name=configuration['dataset'],
                                       model_name=model,
                                       label=configuration['label'],
                                       func=func,
                                       linear_model=configuration['linear_model'],
                                       content=configuration['content'],
                                       level=configuration['level'],
                                       string_convert=configuration['string_convert'],
                                       concat=configuration['concat'],
                                       percent=configuration['percent'])
        all_results.append(results)
        
    return all_results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="model directory")
    args = parser.parse_args()
    percent = None
    
    config = {
              0: {"dataset": "mesa", "label": "nsrr_ahi_hp3r_aasm15", "linear_model": "ridge", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent},
              1: {"dataset": "mesa", "label": "nsrr_ahi_hp4u_aasm15", "linear_model": "ridge", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent},
              2: {"dataset": "ppg-bp", "label": "sysbp", "linear_model": "ridge", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent},
              3: {"dataset": "ppg-bp", "label": "diasbp", "linear_model": "ridge", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent},
              4: {"dataset": "ppg-bp", "label": "hr", "linear_model": "ridge", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent},
              5: {"dataset": "dalia", "label": "hr", "linear_model": "ridge", "level": "segment", "content": "_segment", "string_convert": False, 'concat': False, 'percent': percent},
              6: {"dataset": "numom2b", "label": "ga_at_stdydt", "linear_model": "ridge", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent},
              7: {"dataset": "vv", "label": "bp_sys", "linear_model": "ridge", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent},
              8: {"dataset": "vv", "label": "bp_dia", "linear_model": "ridge", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent},
              }

    all_results = get_results(args.model, config)
    df = pd.DataFrame(all_results)
    df.to_csv(f"../../results/{args.model}_regression_{str(percent)}_{str(int(time()))}.csv", index=False)