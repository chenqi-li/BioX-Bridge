# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np
import pandas as pd
import torch
from .utils import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

def get_data_for_ml(df, dict_embeddings, label):
    y = []
    for key in dict_embeddings.keys():
        y.append(df[df.caseid == key].loc[:, label].values[0])
    X = np.vstack([k.cpu().detach().numpy() if type(k) == torch.Tensor else k for k in dict_embeddings.values()])
    return X, np.array(y)

def regression_model(estimator, param_grid, X_train, y_train, X_test, y_test):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    grid_search = GridSearchCV(estimator=estimator, 
                        param_grid=param_grid, 
                        cv=4, 
                        scoring='neg_mean_squared_error', 
                        verbose=2, 
                        n_jobs=-1)

    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)

    lower_bound_ci_mae, upper_bound_ci_mae, _ = bootstrap_metric_confidence_interval(y_test=y_test,
                                                                                        y_pred=y_pred,
                                                                                        metric_func=mean_absolute_error)

    lower_bound_ci_r2, upper_bound_ci_r2, _ = bootstrap_metric_confidence_interval(y_test=y_test,
                                                                                    y_pred=y_pred,
                                                                                    metric_func=r2_score)          

    results = {'parameters': grid_search.best_params_,
            'mae': mean_absolute_error(y_test, y_pred),
            'lower_bound_mae': lower_bound_ci_mae,
            'upper_bound_mae': upper_bound_ci_mae,
            'r2': r2_score(y_test, y_pred),
            'lower_bound_r2': lower_bound_ci_r2,
            'upper_bound_r2': upper_bound_ci_r2,
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist()}

    return results