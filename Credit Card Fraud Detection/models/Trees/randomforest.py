from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.utils import (
    custom_f1_score,
    load_split_train_splitted,
    load_split_trainval_splitted,
    visualize_all,
    save_experiment,
    calc_scores,
)

"""
Implements a Random Forest Classifier for credit card fraud detection.
Performs grid search for hyperparameters, evaluates, and saves the best model.
"""

# you have the best parameters below
# this search space is huge i rcommend dividing it into 2 or even 3  exp
# random_forest_grid = {
#     "n_estimators": [100, 200,250],
#     "criterion": ["gini", "entropy", "log_loss"],
#     "bootstrap": [True, False],
#     "max_features": [ 1.0,0.5, "sqrt"],  # class 0.4 ,2
#     "class_weight": [{1:2} , {1:4}, {1:8} ,{1:12} ],
#     "max_depth":[6 ,8 ,12] ,
#     "max_leaf_nodes":[ 4 , 8 , 16]
# }


"""
best params from the first trials
{'bootstrap': False, # settled to False
'class_weight': {1: 12},# explore boundary more
'criterion': 'gini', # settled
'max_depth': 6,  # explore boundary more
'max_features': 0.5, # you can take it but i would like to explore it again
'max_leaf_nodes': 16, # explore the boundary
'n_estimators': 200} # explore the boundary
"""


# random_forest_grid = {
#     "n_estimators": [250,350,500],
#     "criterion": ["gini"],
#     "bootstrap": [ False],
#     "max_features": [ 1.0,0.5, "sqrt"],
#     "class_weight": [{1:12} ,{1:16} ,{1:20} ],
#     "max_depth":[4,6 ] ,
#     "max_leaf_nodes":[ 16 , 20 ]
# }
# we got from the second trial
"""
{'bootstrap': False, 
'class_weight': {1: 12},
 'criterion': 'gini', 
 'max_depth': 6, 
 'max_features': 0.5, 
 'max_leaf_nodes': 16,
  'n_estimators': 250 
  }
"""
random_forest_grid = {
    "n_estimators": [250],
    "criterion": ["gini"],
    "bootstrap": [False],
    "max_features": [0.5],
    "class_weight": [{1: 12}],
    "max_depth": [6],
    "max_leaf_nodes": [16],
}
random_forest_clf = RandomForestClassifier(n_jobs=1, random_state=42)

x, y = load_split_train_splitted()
x_val, y_val = load_split_trainval_splitted()
kf = StratifiedKFold(n_splits=2)

if __name__ == "__main__":
    grid = GridSearchCV(
        estimator=random_forest_clf,
        param_grid=random_forest_grid,
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

    visualize_all(grid.best_estimator_, x_val, y_val)

    user_choice = input("do you want to save the model , answer yes|no \n")
    if user_choice.lower() == "yes":
        save_experiment(
            model=grid.best_estimator_,
            params=grid.best_params_,
            metrics=calc_scores(grid.best_estimator_, x_val, y_val),
            model_name="RandomForest",
        )
