"""
Implements an XGBoost Classifier for credit card fraud detection.
Performs grid search for hyperparameters, evaluates, and saves the best model.
"""

import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.utils import (
    custom_f1_score,
    load_split_train_splitted,
    load_split_trainval_splitted,
    visualize_all,
    save_experiment,
    calc_scores,
)
import torch


# Create XGBoost Classifier
xgb_clf = xgb.XGBClassifier(
    n_jobs=1,
    tree_method="hist",
    eval_metric=["logloss", "auc"],
    device="cuda",  # essential if you have a GPU to speed up the training phase
    seed=42,
)

# Load training and validation splits
x, y = load_split_train_splitted()
x_val, y_val = load_split_trainval_splitted()

kf = StratifiedKFold(n_splits=2)
# attention! this is huge search space so split it over 3 or something
# xgb_grid = {
#     "max_depth": [4 , 6, 8],
#     "n_estimators": [2000,5000,7500],
#     "scale_pos_weight": [4 , 5, 10 ],
#     "eta": [0.01, 0.1 ,0.3],
#     "lambda":[1,5,10] ,
#     'gamma':[0,0.25 , 1 ]
# }

# second trial
# xgb_grid = {
#     "max_depth": [ 8 , 12, 16],
#     "n_estimators": [7500,],
#     "scale_pos_weight": [ 10 , 15 , 20],
#     "eta": [0.01],
#     "lambda":[10 , 15 ,20] ,
#     'gamma':[0 ]


# xgb_grid = {
#     "max_depth": [8],
#     "n_estimators": [7500,],
#     "scale_pos_weight": [10],
#     "eta": [0.01],
#     "lambda": [20, 25, 30],
#     "gamma": [0],
# }

# we got
xgb_grid = {
    "max_depth": [8],
    "n_estimators": [7500],
    "scale_pos_weight": [10],
    "eta": [0.01],
    "lambda": [20, 25, 30],
    "gamma": [0],
}

if __name__ == "__main__":
    # Grid search for best hyperparameters
    grid = GridSearchCV(
        estimator=xgb_clf,
        param_grid=xgb_grid,
        n_jobs=-1,
        cv=kf,
        scoring=custom_f1_score,
        verbose=2,
    )
    grid.fit(x, y)

    print("*" * 20)
    print("scoring over  train data is ", calc_scores(grid.best_estimator_, x, y))
    print("best grid parameters are", grid.best_params_)
    print("*" * 20)

    # Visualize performance on validation set
    visualize_all(grid.best_estimator_, x_val, y_val)

    # Ask user if they want to save the model
    user_choice = input("do you want to save the model , answer yes|no \n")
    if user_choice.lower() == "yes":
        save_experiment(
            model=grid.best_estimator_,
            params=grid.best_params_,
            metrics=calc_scores(grid.best_estimator_, x_val, y_val),
            model_name="XGboost",
        )
