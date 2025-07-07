"""
This module implements a voting classifier that combines multiple models built for credit card fraud detection.
It supports both hard and soft voting strategies, evaluates ensemble performance, and saves the best ensemble model.
"""

from sklearn.ensemble import VotingClassifier
from torch.distributed.tensor.debug import visualize_sharding

from src.utils import (
    load_models_eval,  # Loads evaluation results for a list of models
    load_split_trainval_splitted,  # Loads train+val split
    load_split_val_splitted,  # Loads validation split
    calc_scores,  # Calculates evaluation metrics
    visualize_all,  # Visualizes model performance
    load_models,  # Loads model objects
    load_split_train_splitted,  # Loads training split
    load_split_test_splitted,  # Loads test split
    save_experiment,  # Saves experiment results

)

# for loading models
from src.custom_classes import (
    CustomVotingClassifier,
    NNTorch,
    NNWithFocalLoss ,
    LogisticRegressionWithFocalLoss ,
    LogisticRegressionTorch
)

from sklearn.metrics import f1_score
import xgboost, catboost, lightgbm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import f1_score
import numpy as np


# Load data splits
x_train, y_train = load_split_train_splitted()
x_train_val, y_train_val = load_split_trainval_splitted()
x_val, y_val = load_split_val_splitted()


# List of model names to include in the ensemble
model_names = [
    "adaboost",
    "decisiontree",
    "gradientboost",
    "lgbmboost",
    "LogisticRegression",
    "randomforest",
    "xgboost",
    "xgboostrf",
    "logisticregressionwithfocalloss",
    "knn" ,
    "NNWithFocalLoss"
]
# model_score_trainval = load_models_eval(models_names=model_names, x=x_train_val, y=y_train_val, score=f1_score)

# Evaluate models on validation data and extract their scores
model_score_val = load_models_eval(
    models_names=model_names, x=x_val, y=y_val, scores=f1_score
)
scores = [item[-1] for item in model_score_val]

# Load model objects
loaded_models = load_models(model_names)
str_models = list(zip(model_names, loaded_models))

# Compute weights for soft voting based on validation scores
weights = np.array(scores) / sum(scores)


# print(weights)


if __name__ == "__main__":
    # print("model over the train val", [item[-1] for item in model_score_trainval])
    # print("model over the  val",scores)

    # voting_soft_clf = CustomVotingClassifier(estimators=str_models, weights=weights, voting="soft")

    # Example: Hard voting ensemble
    voting_hard_clf = CustomVotingClassifier(
        estimators=str_models, weights=weights, voting="hard"
    )  # Hard voting is slightly better over the val

    visualize_all(voting_hard_clf , x_train ,y_train)
    visualize_all(voting_hard_clf, x_train_val, y_train_val)


    print("--/" * 20)
    print("voting classifer over the  train")
    print(
        "f1 hard voting",
        f1_score(y_true=y_train, y_pred=voting_hard_clf.predict(x_train)),
    )
    print("--/" * 20)
    print("voting classifer over the val train")

    print(
        "f1 hard voting",
        f1_score(y_true=y_train_val, y_pred=voting_hard_clf.predict(x_train_val)),
    )
    # print("f1 soft voting", f1_score(y_true=y_train_val, y_pred=voting_soft_clf.predict(x_train_val)))

    print("voting classifer over the val ")
    print(
        "f1 hard voting", f1_score(y_true=y_val, y_pred=voting_hard_clf.predict(x_val))
    )
    # print("f1 soft voting", f1_score(y_true=y_val, y_pred=voting_soft_clf.predict(x_val)))

    # print(calc_scores(voting_clf , x_train_val ,y_train_val))
    print("--/" * 20)
    # visualize_all(voting_clf , x_val , y_val) # this is  not avilable for hard classifier


    # Save the hard voting experiment results
    save_experiment(
        model=voting_hard_clf,
        params={"voting": "hard"},
        metrics={
            "f1 score": f1_score(y_true=y_val, y_pred=voting_hard_clf.predict(x_val)),
            "average precision": None,
            "area_under_pr_curve": None,
            "area under roc": None,
        },
        model_name="HardVoting",
    )
