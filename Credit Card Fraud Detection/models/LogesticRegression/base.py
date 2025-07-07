from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import make_scorer, f1_score
from src.utils import load_split_train_splitted, load_split_trainval_splitted
import numpy as np
from src.utils import save_experiment, calc_scores, visualize_all, custom_f1_score


x, y = load_split_train_splitted()

x_val, y_val = load_split_trainval_splitted()


model = LogisticRegression(solver="lbfgs", max_iter=200_000, random_state=42, verbose=1, n_jobs=1)


# some models recommend ratio (major class/ minor class ) for the weight
fraction = np.sum(y) / (x.shape[0] - np.sum(y))

# first tial we got C:0.5 , class_weight {1:4} ,fit_intercept = True
# param_grid = {"class_weight": [{1: w} for w in [4, 8, 10, 15, fraction / 2]],
#               "C": [0.25, 0.5, 1], "fit_intercept": [True, False]
#               }

# second trial we got C =0.4 , class_weight = 3 ,
# param_grid = {"class_weight": [{1: w} for w in [ 2 , 3 ,4 ]],
#               "C": [0.4, 0.5, 0.6],
#               "fit_intercept": [True]
#               }

# thid tiral we got C =0.3 and class_weight =3
# param_grid = {"class_weight": [{1: w} for w in [ 1,2 , 3  ]],
#               "C": [0.1 , 0.2 ,0.3 ,0.4],
#               "fit_intercept": [True]
#               }

# final check to
param_grid = {"class_weight": [{1: w} for w in [3, 4, 5, 10, 12, 15]], "C": [0.3], "fit_intercept": [True]}
kf = StratifiedKFold(n_splits=2, shuffle=False)

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, n_jobs=-1, scoring=custom_f1_score, refit=True)

grid.fit(x, y)


save_experiment(
    model=grid.best_estimator_,
    params=grid.best_params_,
    metrics=calc_scores(grid.best_estimator_, x_val, y_val),
    model_name="LogisticRegression",
)


if __name__ == "__main__":
    print("*" * 20)
    print("scoring over  train data is ", calc_scores(grid.best_estimator_, x, y))
    print("best grid parameters are", grid.best_params_)
    visualize_all(grid.best_estimator_, x_val, y_val)
