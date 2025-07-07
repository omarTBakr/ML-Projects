"""
Implements a Gradient Boosting Classifier for credit card fraud detection.
Performs grid search for hyperparameters, evaluates, and saves the best model.
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.utils import (
    custom_f1_score,
    load_split_train_splitted,
    load_split_trainval_splitted,
    visualize_all,
    save_experiment,
    calc_scores,
)


# Create Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(random_state=42)

# Load training and validation splits
x, y = load_split_train_splitted()
x_val, y_val = load_split_trainval_splitted()


# kf = StratifiedKFold(n_splits=2) # for the first trial
######attention , this search space is huge , you may want to divide it############33
# adaboost_grid = {
#     "n_estimators": [100, 200 , 500],
#     "loss":["log_loss" ,"exponential"],
#     "learning_rate": [0.1, 0.5],
#     "max_depth":[3,6,8] ,
#     "max_features":[1.0 , 0.5 , "sqrt"] ,
#
# }
# we had 'learning_rate': 0.5, 'loss': 'exponential', 'max_depth': 3, 'max_features': 'sqrt', 'n_estimators': 500}
# which means we need to explore more estimators , but i will reduce the max_depth to regularize the tree to reduce over fitting
# increase fold to reduce overfitting also
kf = StratifiedKFold(n_splits=3)
# adaboost_grid = {
#     "n_estimators": [500,750,1000],
#     "loss":["exponential"],
#     "learning_rate": [0.1 ,0.5],
#     "max_depth":[2,3] ,
#     "max_features":["sqrt"] ,
#
# }
# we got form secon trial {'learning_rate': 0.1, 'loss': 'exponential', 'max_depth': 3, 'max_features': 'sqrt', 'n_estimators': 500}
adaboost_grid = {
    "n_estimators": [500],
    "loss": ["exponential"],
    "learning_rate": [0.1],
    "max_depth": [3],
    "max_features": ["sqrt"],
}

if __name__ == "__main__":
    # Grid search for best hyperparameters
    grid = GridSearchCV(
        estimator=gb_clf,
        param_grid=adaboost_grid,
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
            model_name="GradientBoost",
        )
