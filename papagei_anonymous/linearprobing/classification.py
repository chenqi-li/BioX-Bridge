# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np
import pandas as pd
import torch
import json
from utils import bootstrap_metric_confidence_interval
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def classification_model(estimator, param_grid, X_train, y_train, X_test, y_test, bootstrap=True):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    grid_search = GridSearchCV(estimator=estimator, 
                        param_grid=param_grid, 
                        cv=4, 
                        scoring='accuracy', 
                        verbose=2, 
                        n_jobs=-1)

    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
    
    lower_bound_ci_auc, upper_bound_ci_auc, lower_bound_ci_f1, upper_bound_ci_f1 = -999, -999, -999, -999 
    if bootstrap:

        lower_bound_ci_auc, upper_bound_ci_auc, _ = bootstrap_metric_confidence_interval(y_test=y_test,
                                                                                                y_pred=y_pred_proba,
                                                                                                metric_func=roc_auc_score)

        lower_bound_ci_f1, upper_bound_ci_f1, _ = bootstrap_metric_confidence_interval(y_test=y_test,
                                                                                        y_pred=y_pred,
                                                                                        metric_func=f1_score)                                                                                     

    results = {'parameters': grid_search.best_params_,
            'auc': roc_auc_score(y_test, y_pred_proba),
            'auc_lower_ci': lower_bound_ci_auc,
            'auc_upper_ci': upper_bound_ci_auc,
            'f1': f1_score(y_test, y_pred),
            'f1_lower_ci': lower_bound_ci_f1,
            'f1_upper_ci': upper_bound_ci_f1,
            'y_test': y_test,
            'y_pred_proba':json.dumps(y_pred_proba.tolist()),
            'y_pred': json.dumps(y_pred.tolist())}

    return results