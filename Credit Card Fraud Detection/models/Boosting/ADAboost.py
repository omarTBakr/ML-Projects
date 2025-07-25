"""
Implements an AdaBoost Classifier for credit card fraud detection.
Performs grid search for hyperparameters, evaluates, and saves the best model.
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.utils import (
    custom_f1_score,
    load_split_train_splitted,
    load_split_trainval_splitted,
    visualize_all,
    save_experiment,
    calc_scores,
)


# Create AdaBoost Classifier
ada_boost_clf = AdaBoostClassifier(random_state=42)

# Load training and validation splits
x, y = load_split_train_splitted()
x_val, y_val = load_split_trainval_splitted()


kf = StratifiedKFold(n_splits=2)
# adaboost_grid = {
#     "n_estimators": [1000, 1500, 2000],
#     "learning_rate": [ 0.05,0.1],
# }

# we got form the first trial {'learning_rate': 0.1, 'n_estimators': 2000}
# which means that we have to check boundaries also

# we got {'learning_rate': 0.5, 'n_estimators': 5000} from the second trail
# which indicates that we still need to check boundaries
# adaboost_grid = {
#     "n_estimators": [ 2000,3000,5000],
#     "learning_rate": [ 0.1 ,0.2 ,0.5],
# }

# we have found  {'learning_rate': 0.5, 'n_estimators': 5000}
# adaboost_grid = {
#     "n_estimators": [ 5000 ,7500],
#     "learning_rate": [ 0.5 , 0.6],
# }
adaboost_grid = {
    "n_estimators": [5000],
    "learning_rate": [0.5],
}
if __name__ == "__main__":
    # Grid search for best hyperparameters
    grid = GridSearchCV(
        estimator=ada_boost_clf,
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
            model_name="AdaBoost",
        )
