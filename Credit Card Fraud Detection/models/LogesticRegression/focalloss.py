import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils import load_split_train_splitted, load_split_trainval_splitted, save_experiment, custom_f1_score
from torch import nn

import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torchvision.ops import sigmoid_focal_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.utils import visualize_all, calc_scores
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from src.custom_classes import  LogisticRegressionTorch , LogisticRegressionWithFocalLoss



if __name__ == "__main__":
    # leave scaling to pipeline
    x, y = load_split_train_splitted()
    # x, scaler = scale_date(x, scaler=None)
    x_val, y_val = load_split_trainval_splitted()
    # x_val,_ = scale_date(x_val , scaler)

    model = LogisticRegressionWithFocalLoss(n_epochs=350)
    pipe = Pipeline([("scaler", MinMaxScaler()), ("model", model)])
    # first tiral we got {'alpha': 0.5, 'gamma': 2, 'weight_decay': 0}
    # param_grid = {"alpha": [0.25, 0.5, 0.75],
    #               "gamma": [ 1, 2, 5],
    #               "weight_decay": [0, 0.001, 0.01, 0.1]
    #               }
    # scond trial we have got  {'alpha': 0.5, 'gamma': 4, 'weight_decay': 0}
    # but wose perforamcne than the first so we will roll back to testing alpha
    # with the new gamma
    # param_grid = {"alpha": [0.4, 0.5, 0.6],
    #               "gamma": [ 2,3,4],
    #               "weight_decay": [0, 0.001, 0.01, 0.1]
    #               }

    # we got {'alpha': 0.3, 'gamma': 4, 'weight_decay': 0}
    param_grid = {"alpha": [0.3, 0.4, 0.5, 0.6], "gamma": [4], "weight_decay": [0]}
    kf = StratifiedKFold(n_splits=2, shuffle=False)

    grid = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=kf, n_jobs=-1, scoring=custom_f1_score, refit=True, verbose=2
    )

    grid.fit(x, y)

    print("*" * 20)
    print("scoring over  train data is ", calc_scores(grid.best_estimator_, x, y))
    print("best grid parameters are", grid.best_params_)
    print("*" * 20)

    visualize_all(grid.best_estimator_, x_val, y_val)

    save_experiment(
        model=grid.best_estimator_,
        params=grid.best_params_,
        metrics=calc_scores(grid.best_estimator_, x_val, y_val),
        model_name="LogisticRegressionWithFocalLoss",
    )
