"""
Implements an XGBoost Random Forest Classifier for credit card fraud detection.
Performs grid search for hyperparameters, evaluates, and saves the best model.

- this is same as Xgboost but used random forest classifier as a base estimator
    instead of xgboost tree

- it happens to give a slightly better performance that xgboost base classifier
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


# Create XGBRF Classifier
xgbrf_clf = xgb.XGBRFClassifier(
    n_jobs=1,
    tree_method="hist",
    eval_metric=["logloss"],
    objective="binary:logistic",
    # device ='cuda',  # Uncomment if GPU is available
    seed=42,
)

# Load training and validation splits
x, y = load_split_train_splitted()
x_val, y_val = load_split_trainval_splitted()

kf = StratifiedKFold(n_splits=2)
# attention! this is huge search space so split it over 3 or something
# xgbrf_grid = {
#     "n_estimators": [2000, 5000],
#     "max_depth": [4 , 6],
#     "scale_pos_weight": [4 , 5],
#     "learning_rate": [0.5, 1 ],
#     "reg_lambda":[1,5] ,
#     'gamma':[0,0.25 ]
# }

# we got {'gamma': 0.25, 'learning_rate': 1, 'max_depth': 4, 'n_estimators': 5000, 'reg_lambda': 5, 'scale_pos_weight': 4}
# which indicated that we need to for more estimators and more reg

# xgbrf_grid = {
#     "n_estimators": [ 5000,7500],
#     "max_depth": [4 ,6],
#     "scale_pos_weight": [4 ,5],
#     "learning_rate": [  0.5,1 ],
#     "reg_lambda":[5 ,6] ,
#     'gamma':[0,0.25 ,0.3]
# }
# {'gamma': 0.25, 'learning_rate': 1, 'max_depth': 4, 'n_estimators': 5000, 'reg_lambda': 5, 'scale_pos_weight': 4}
xgbrf_grid = {
    "n_estimators": [7500],
    "max_depth": [4],
    "scale_pos_weight": [4],
    "learning_rate": [1],
    "reg_lambda": [1],
    "gamma": [0],
}

if __name__ == "__main__":
    # Grid search for best hyperparameters
    grid = GridSearchCV(
        estimator=xgbrf_clf,
        param_grid=xgbrf_grid,
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
            model_name="XGboostRf",
        )
