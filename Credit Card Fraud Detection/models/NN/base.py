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
from src.custom_classes import   NNTorch ,NNWithFocalLoss



if __name__ == "__main__":
    # leave scaling to pipeline
    x, y = load_split_train_splitted()
    # x, scaler = scale_date(x, scaler=None)
    x_val, y_val = load_split_trainval_splitted()
    # x_val,_ = scale_date(x_val , scaler)

    model = NNWithFocalLoss(n_epochs=400)
    pipe = Pipeline([("scaler", MinMaxScaler()), ("model", model)])

    # param_grid = {"alpha": [0.25, 0.5, 0.75],
    #               "gamma": [ 1, 2, 5],
    #               "weight_decay": [0, 0.001, 0.01, 0.1]
    #               }

    # from the first trial {'alpha': 0.25, 'gamma': 2, 'weight_decay': 0}



    # param_grid = {"alpha": [0.1,0.25,0.3],
    #               "gamma": [ 1,2,3],
    #               "weight_decay": [0]
    #               }
    # we got {'alpha': 0.3, 'gamma': 2, 'weight_decay': 0}
    param_grid = {"alpha": [0.3],
                  "gamma": [ 2],
                  "weight_decay": [0]
                  }
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

    if input('Do you want to save the model yes|no ') =='yes':
        save_experiment(
            model=grid.best_estimator_,
            params=grid.best_params_,
            metrics=calc_scores(grid.best_estimator_, x_val, y_val),
            model_name="LogisticRegressionWithFocalLoss",
        )
