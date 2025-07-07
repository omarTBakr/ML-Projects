"""
Implements a LightGBM Classifier for credit card fraud detection.
Performs grid search for hyperparameters, uses early stopping, evaluates, and saves the best model.
"""

from lightgbm import LGBMClassifier
import lightgbm
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.utils import (
    custom_f1_score,
    load_split_train_splitted,
    load_split_trainval_splitted,
    visualize_all,
    save_experiment,
    calc_scores,
)


# Load training and validation splits
x, y = load_split_train_splitted()
x_val, y_val = load_split_trainval_splitted()

# Create LightGBM Classifier
lgbm_clf = LGBMClassifier(random_state=42, n_jobs=1)

# Define parameter grid for grid search (final trial)
lgmb_grid = {
    "n_estimators": [750],
    "max_depth": [-1],  # this is settled
    "scale_pos_weight": [5],
    "reg_lambda": [20],
    "learning_rate": [0.1],  # i think 0.1 is good but let's try anyway
}

# Early stopping parameters for LightGBM
fit_params = {
    "eval_set": [(x_val, y_val)],  # The dataset to monitor
    "callbacks": [lightgbm.early_stopping(stopping_rounds=50, verbose=False)],
}

# we got {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 500, 'reg_lambda': 10, 'scale_pos_weight': 5}
# which means that we should try more estimators  , and try bigger values for lambda

# lgmb_grid = {
#     "n_estimators": [750,1000],
#      "max_depth": [6, 9, -1],
#     "scale_pos_weight": [4, 5, 10],
#     "reg_lambda": [10 ,12 , 15],
#     "learning_rate": [0.1, 0.3, 0.5],
# }
# got {'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 750, 'reg_lambda': 15, 'scale_pos_weight': 5}
#  i think we can use more estimator which is definately increase f score but use higher vlaues for lambda

# lgmb_grid = {
#     "n_estimators": [750 ,1000 ],
#      "max_depth": [6, 9, -1],
#     "scale_pos_weight": [4, 5, 10],
#     "reg_lambda": [ 15 , 20 ,30 ],
#     "learning_rate": [0.1, 0.3, 0.5],
# }
# we got {'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 750, 'reg_lambda': 20, 'scale_pos_weight': 5}
# lgmb_grid = {
#     "n_estimators": [750 ,1000 ],
#      "max_depth": [-1], # this is settled
#     "scale_pos_weight": [5 ,7 ,9],
#     "reg_lambda": [ 20 ,25,30 ],
#     "learning_rate": [0.1, 0.3, 0.5], # i think 0.1 is good but let's try anyway
# }
# we got {'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 750, 'reg_lambda': 20, 'scale_pos_weight': 5} which is exactly the above

kf = StratifiedKFold(n_splits=2)

if __name__ == "__main__":
    # Grid search for best hyperparameters
    grid = GridSearchCV(
        estimator=lgbm_clf,
        param_grid=lgmb_grid,
        n_jobs=-1,
        cv=kf,
        scoring=custom_f1_score,
        verbose=2,
    )
    grid.fit(x, y, **fit_params)

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
            model_name="LGBMboost",
        )
