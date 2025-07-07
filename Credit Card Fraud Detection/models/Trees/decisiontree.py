"""
Implements a Decision Tree Classifier for credit card fraud detection.
Performs grid search for hyperparameters, evaluates, and saves the best model.
"""

from sklearn import tree
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.utils import (
    custom_f1_score,
    load_split_train_splitted,
    load_split_trainval_splitted,
    visualize_all,
    save_experiment,
    calc_scores,
)

x, y = load_split_train_splitted()
x_val, y_val = load_split_trainval_splitted()

# tree_grid = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [3, 5, 7, 10],
#     'min_samples_leaf': [10, 20, 50],
#     'min_samples_split': [20, 40, 100],
#     'class_weight': ['balanced', {1: 25}, {1: 50}] }

# we got {'class_weight': {1: 50}, 'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 10, 'min_samples_split': 20} form first tiral
# tree_grid = {
#     'criterion': [ 'entropy'],
#     'max_depth': [3, 5, 7, 10],
#     'min_samples_leaf': [8,9,10 ],
#     'min_samples_split': [15,20],
#     'class_weight': [ {1: 50} , {1:65} ,{1:100}] }

# we got {'class_weight': {1: 50}, 'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 8, 'min_samples_split': 20} which is pretty much what we had above which means
# that these are the best parameters for this problem

# Define parameter grid for grid search (final trial)
tree_grid = {
    "criterion": ["entropy"],
    "max_depth": [3],
    "min_samples_leaf": [8],
    "min_samples_split": [20],
    "class_weight": [{1: 50}],
}

# Create Decision Tree Classifier
tree_clf = tree.DecisionTreeClassifier()
kf = StratifiedKFold(n_splits=2)

if __name__ == "__main__":
    # Grid search for best hyperparameters
    grid = GridSearchCV(
        estimator=tree_clf,
        param_grid=tree_grid,
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
            model_name="DecisionTree",
        )
