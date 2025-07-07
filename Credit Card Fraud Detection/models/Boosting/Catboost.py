"""
Implements a CatBoost Classifier for credit card fraud detection.
Performs grid search for hyperparameters, evaluates, and saves the best model.
"""

from catboost import CatBoostClassifier
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

kf = StratifiedKFold(n_splits=2)

# Create CatBoost Classifier
cat_clf = CatBoostClassifier(
    # eval_metric=["F1", "AUC", "logloss"] ,
    #       task_type="GPU",# do this for GPU Training but i don't think it like the eval_metric
    #       devices='0'
)

# Define parameter grid for grid search (final trial)
cat_grid = {
    "max_depth": [6, 8, 10],
    "n_estimators": [5000],
    "scale_pos_weight": [10, 12],
    "eta": [0.3, 0.4],
    "reg_lambda": [10, 12],
}

if __name__ == "__main__":
    # Grid search for best hyperparameters
    grid = GridSearchCV(
        estimator=cat_clf,
        param_grid=cat_grid,
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
