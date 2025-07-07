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
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import  MinMaxScaler

x_train , y_train = load_split_train_splitted()
x_val , y_val = load_split_trainval_splitted()

# knn_grid = {
#     'estimator__weights':['uniform', 'distance'] ,
#     'estimator__n_neighbors':[2,5,10 ] ,
#     'estimator__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'] ,
#     'estimator__leaf_size':[ 3 ,5 ,10]
#
# }
# we got {'estimator__algorithm': 'auto', 'estimator__leaf_size': 3,
# 'estimator__n_neighbors': 5, 'estimator__weights': 'distance'}
knn_grid = {
    'estimator__weights':['distance'] ,
    'estimator__n_neighbors':[3,5,9 ] ,
    'estimator__algorithm':['auto'] ,
    'estimator__leaf_size':[ 2,3 ,5 ]

}


knn= KNeighborsClassifier(n_jobs=1)

pipe = Pipeline(
    [('scaler',MinMaxScaler())
    ,('estimator' , knn)]
)
kf = StratifiedKFold(n_splits=5)

if __name__ =='__main__':

    grid =GridSearchCV( estimator= pipe ,
                        scoring = custom_f1_score,
                        param_grid=knn_grid ,
                        cv = kf ,
                        n_jobs=-1,
                        verbose=2
                    )
    grid.fit(x_train,y_train)
    print("/*\\" * 20)
    print("scoring over  train data is ", calc_scores(grid.best_estimator_, x_train, y_train))
    print("best grid parameters are", grid.best_params_)
    print("/*\\" * 20)

    # Visualize performance on validation set
    visualize_all(grid.best_estimator_, x_val, y_val)

    # Ask user if they want to save the model
    user_choice = input("do you want to save the model , answer yes|no \n")
    if user_choice.lower() == "yes":
        save_experiment(
            model=grid.best_estimator_,
            params=grid.best_params_,
            metrics=calc_scores(grid.best_estimator_, x_val, y_val),
            model_name="Knn",
        )