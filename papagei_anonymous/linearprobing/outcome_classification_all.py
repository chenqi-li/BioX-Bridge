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
from sklearn.ensemble import RandomForestClassifier
from time import time

def binary_classification(dataset_name, model_name, linear_model, label, func, content, concat, level="patient", string_convert=True, percent=None):

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
        size = int(len(X_train))
        idx = np.random.choice(np.arange(size), size=int(percent * size), replace=False)  # Add replace=False if necessary
        print(f"Selected indices: {idx}")
        
        # Ensure X_train and y_train are numpy arrays
        X_train = np.array(X_train)[idx.astype(int)]
        y_train = np.array(y_train)[idx.astype(int)]
        print(f"Using {len(X_train)} of {size}")

    # Define the parameter grid
    if linear_model == "lr":
        estimator = LogisticRegression()
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs'],
            'max_iter': [100, 200]
        }

    if linear_model == "rf":
        estimator = RandomForestClassifier()
        param_grid = {
            'n_estimators': [100, 200],    
            'max_features': ['sqrt', 'log2'],
            'max_depth': [10, 20, 30],     
            'min_samples_split': [2, 5],     
            'min_samples_leaf': [1, 2],      
            }

    results = classification_model(estimator=estimator,
                                param_grid=param_grid,
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test,
                                bootstrap=True)

    results['test_keys'] = test_keys
    results['model'] = model_name
    results['dataset'] = dataset_name
    results['label'] = label
    
    return results

def multilabel_classification(dataset_name, model_name, label, func, content, concat, level="patient", string_convert=True):


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
    
    estimator = RandomForestClassifier()
    param_grid = {
        'n_estimators': [100, 200],    
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30],     
        'min_samples_split': [2, 5],     
        'min_samples_leaf': [1, 2],      
        }
                                           
                                                                                     
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    grid_search = GridSearchCV(estimator=estimator, 
                        param_grid=param_grid, 
                        cv=StratifiedKFold(n_splits=3), 
                        scoring='accuracy', 
                        verbose=2, 
                        n_jobs=-1)

    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    
    lower_bound_ci_acc, upper_bound_ci_acc, _ = bootstrap_metric_confidence_interval(y_test=np.array(y_test),
                                                                                    y_pred=np.array(y_pred),
                                                                                    metric_func=accuracy_score)     
    
    results = {'parameters': grid_search.best_params_,
        'acc': accuracy_score(y_test, y_pred),
        'acc_lower_ci': lower_bound_ci_acc,
        'acc_upper_ci': upper_bound_ci_acc,
        'y_test': y_test,
        'y_pred': y_pred,
        'test_keys': test_keys,
        'model': model_name}
    
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
        if configuration['classification_type'] == "binary":
            results = binary_classification(dataset_name=configuration['dataset'],
                                           model_name=model,
                                           linear_model=configuration['linear_model'],
                                           label=configuration['label'],
                                           func=func,
                                           content=configuration['content'],
                                           level=configuration['level'],
                                           string_convert=configuration['string_convert'],
                                           concat=configuration['concat'],
                                           percent=configuration['percent'])
            
        if configuration['classification_type'] == "multi":
            results = multilabel_classification(dataset_name=configuration['dataset'],
                                           model_name=model,
                                           label=configuration['label'],
                                           func=func,
                                           content=configuration['content'],
                                           level=configuration['level'],
                                           string_convert=configuration['string_convert'],
                                           concat=configuration['concat'])
        all_results.append(results)
        
    return all_results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="model directory")
    parser.add_argument('classification_type', type=str)
    args = parser.parse_args()
    percent = None

    if args.classification_type == "binary":
        config = {
            0: {"dataset": "sdb", "label": "AHI", "classification_type": "binary", "linear_model": "lr", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent},
            1: {"dataset": "vital", "label": "icu_days", "classification_type": "binary", "linear_model": "lr", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent},
            2: {"dataset": "mesa", "label": "nsrr_ever_smoker", "classification_type": "binary", "linear_model": "lr", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent},
            3: {"dataset": "mimic", "label": "DOD", "classification_type": "binary", "linear_model": "lr", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent},
            4: {"dataset": "ppg-bp", "label": "Hypertension", "classification_type": "binary", "linear_model": "lr", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent},
            5: {"dataset": "wesad", "label": "valence", "classification_type": "binary", "linear_model": "lr", "level": "segment", "content": "_segment", "string_convert": False, 'concat': False, 'percent': percent}, 
            6: {"dataset": "wesad", "label": "arousal", "classification_type": "binary", "linear_model": "lr", "level": "segment", "content": "_segment", "string_convert": False, 'concat': False, 'percent': percent}, 
            7: {"dataset": "ecsmp", "label": "TMD", "classification_type": "binary", "linear_model": "lr", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent}, 
            # 8: {"dataset": "ecsmp", "label": "sds", "classification_type": "binary", "linear_model": "lr", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent}, 
            9: {"dataset": "numom2b", "label": "stdyvis", "classification_type": "binary", "linear_model": "lr", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent},
            }

    if args.classification_type == "multi":   
        config = {
            0: {"dataset": "vital", "label": "optype", "classification_type": "multi", "linear_model": "rf", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False},
            1: {"dataset": "dalia", "label": "activity", "classification_type": "multi", "linear_model": "rf", "level": "segment", "content": "_segment", "string_convert": False, 'concat': False},
            2: {"dataset": "wesad", "label": "affect", "classification_type": "multi", "linear_model": "rf", "level": "segment", "content": "_segment", "string_convert": False, 'concat': False},   
        }

    all_results = get_results(args.model, config)
    df = pd.DataFrame(all_results)
    # df.to_csv(f"../../results/{args.model}_{args.classification_type}_{str(percent)}_{str(int(time()))}.csv", index=False)
    df.to_csv(f"../../results/{args.model}_{args.classification_type}_{str(int(time()))}.csv", index=False)