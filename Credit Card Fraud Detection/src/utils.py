"""
this file will contain all utitls needed for data like
- loading
- preprocessing

"""


from src.path_config import SPLITED_DATA_DIR, KAGGLE_DATA_PATH, EXP_DIR , FIGURES_DIR
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    auc,)
import  sklearn



from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import json
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, f1_score
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from src.custom_classes import (LogisticRegressionTorch ,
                                LogisticRegressionWithFocalLoss,
                                CustomVotingClassifier ,
                                NNWithFocalLoss,
                                NNTorch)
import  xgboost , catboost



sns.set_theme(style="white")
sns.set_palette("hot")
sns.set(rc={"figure.dpi": 600})

# to avoid warnings  and erros during gridsearch
custom_f1_score = make_scorer(
    f1_score,
    response_method="predict",
    greater_is_better=True,
    pos_label=1,
    zero_division=0,
)
import re


# note RAW_DATA_DIR , KAGGLE_DATA_PATH are path objects form path lib
# that's why we can / to join paths


#########################loading data#################################
def _load_df(path):
    return pd.read_csv(path)


def load_split_train():
    return _load_df(SPLITED_DATA_DIR / "train.csv")


def load_split_train_splitted():
    data = load_split_train()
    return data.drop("Class", axis=1), data["Class"]


def scale_date(data, scaler: MinMaxScaler = None):
    if type(data) == pd.DataFrame:
        data = data.to_numpy()

    if scaler is None:
        scaler = MinMaxScaler().fit(data)

    return scaler.transform(data), scaler


def load_split_trainval():
    return _load_df(SPLITED_DATA_DIR / "trainval.csv")


def load_split_trainval_splitted():
    data = load_split_trainval()
    return data.drop("Class", axis=1), data["Class"]


def load_split_val():
    return _load_df(SPLITED_DATA_DIR / "val.csv")


def load_split_val_splitted():
    data = load_split_val()
    return data.drop("Class", axis=1), data["Class"]


def load_split_test():

    return _load_df(SPLITED_DATA_DIR / "test.csv")


def load_split_test_splitted():
    data = load_split_test()
    return data.drop("Class", axis=1), data["Class"]


def load_kaggle():
    return _load_df(KAGGLE_DATA_PATH)


#################################calculate scores############################
def calc_scores(model, x, y):
    """

    :param model: object that inheritsf rom sklear.base.BaseEstimator and has predict, predict_proba_
    :param x:  data you want to calcuate on
    :param y:  your target variable
    :return:  f1 score , auc  , avg_precision
    """
    y_true, y_pred, y_prop = y, model.predict(x), model.predict_proba(x)[:, 1]

    fscore = f1_score(y_true, y_pred)

    avg_precision = average_precision_score(y_true, y_prop)

    precision, recall, threshold = precision_recall_curve(y_true, y_prop)
    area_precision_recall = auc(recall, precision)
    fpr, tpr, threshold = roc_curve(y_true, y_pred)

    roc_area = auc(fpr, tpr)

    return {
        "f1 score": fscore,
        "average precision": avg_precision,
        "area_under_pr_curve": area_precision_recall,
        "area under roc": roc_area,
    }


############################# visualize precision, recall#######################
"""
    for the below functions for visulalization 
    model: sklearn modlel or inherits  sklearn.base.BaseEstimator aka  supports 
        .fit  
        .predict
        .predict_proba
    to be abel to calculate the precision, recall, f1 , auc scores 

"""


def vis_pr_rec_curve(model, x, y, ax):
    y_true, y_pred, y_prop = y, model.predict(x), model.predict_proba(x)[:, 1]
    precision, recall, _ = precision_recall_curve(y_true, y_prop)

    ax.plot(recall, precision)
    ax.fill_between(recall, precision, alpha=0.5)

    avg_precision = average_precision_score(y_true, y_prop)
    fscore = f1_score(y_true, y_pred)
    area = auc(recall, precision)

    ax.set_title(f"PR Curve | Avg Precision={avg_precision:.2f} | F1={fscore:.2f} | AUC={area:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")


def vis_roc(model, x, y, ax):
    y_true, y_prop = y, model.predict_proba(x)[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_prop)
    area = auc(fpr, tpr)

    # Use the 'label' kwarg for the legend
    ax.plot(fpr, tpr, label=f"ROC (AUC = {area:.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")

    ax.legend()

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve")

    ax.fill_between(fpr, tpr, alpha=0.3)


def vis_conf_matrix(model, x, y, ax):
    y_true, y_pred = y, model.predict(x)
    matrix = confusion_matrix(y_true, y_pred)

    sns.heatmap(matrix, annot=True, ax=ax, fmt="d", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")


def visualize_all(model, x, y ,
                  general_title = False ,
                  title =None,
                  save = False ):



    fig, axs = plt.subplots(3, 1, figsize=(8, 15))

    vis_pr_rec_curve(model, x, y, axs[0])
    vis_roc(model, x, y, axs[1])

    vis_conf_matrix(model, x, y, axs[2])
    if general_title:
        fig.suptitle(title, fontsize=20, y=0.97)
    if save:
        plt.savefig(f'{FIGURES_DIR/title}.png')

    if not save:
        plt.show()



####################saving and loading the model#################


def save_experiment(model, params, metrics, model_name):
    """
    Saves the model, its parameters, and performance metrics.

    Args:
        model: The trained model object.
        params: Dictionary of parameters used to train the model.
        metrics: Dictionary of performance metrics (e.g., {'f1_score': 0.85}).
        model_name: A descriptive name for the model run (e.g., "xgb_imbalanced_data").
    """
    # Create a unique filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{model_name}_{timestamp}"

    # --- Save the Model ---
    model_path = EXP_DIR / f"{filename_base}.joblib"
    joblib.dump(model, model_path)

    # print(f"Model saved to: {model_path}")

    # --- Save the Configuration and Results ---
    config_path = EXP_DIR / f"{filename_base}.json"

    # Create a dictionary to hold all the info
    all_info = {
        "model_name": model_name,
        "model_class": str(model.__class__.__name__),
        "saved_at": timestamp,
        "model_path": str(model_path),
        "parameters": params,
        "metrics": metrics,
    }

    with open(config_path, "w") as f:
        json.dump(all_info, f, indent=4)

    # print(f"Configuration and metrics saved to: {config_path}")
    print("*" * 20)
    print("Model saved successfuly")
    print("*" * 20)


def load_model(model_name: str, load_metric=False):
    """
    this function will search the experiment directory with the given model and return the latest one
    :param model_name: model name
    :return:
    """

    pattern = re.compile(f"^{re.escape(model_name)}_\\d+.*\\.joblib$", re.IGNORECASE)

    model_files = [f for f in EXP_DIR.glob("**/*.joblib") if pattern.match(f.name)]

    # model_files = list(EXP_DIR.rglob(f"*{model_name}^\d*.joblib", case_sensitive=False))

    if not model_files:
        raise FileNotFoundError(f"Warning: No models found matching the name '{model_name}' in {EXP_DIR}")

    # get th lates file which is the most recent modified
    latest_model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading latest model: {latest_model_path.name}")

    loaded_model = joblib.load(latest_model_path)

    if load_metric:
        latest_metric_path = latest_model_path.stem + ".json"
        return loaded_model, json.load(latest_metric_path).get("metrics", None).get("f1_score", None)

    return loaded_model


def load_models_eval(models_names: list, x, y, scores: sklearn.metrics):
    loaded_models = []
    for model_name in models_names:
        model = load_model(model_name, load_metric=False)
        y_pred = model.predict(x).reshape(-1,1)


        final_score = scores(y_true=y, y_pred=y_pred)
        loaded_models.append((model, final_score))

    return loaded_models


def load_models(models_names: list):
    return [load_model(model_name) for model_name in models_names]



#####################save visulizations for all models##########################
def save_all_visualizations():
    '''
    this function will iterate over modles one by one and save it's visualization
    i forgot to save the figure while experimenting
    :return:
    '''



    x_test , y_test  = load_split_test_splitted()
    for model_name in EXP_DIR.iterdir():
        print(model_name , "it's type is", type(model_name))
        name = model_name.stem.split('_')[0]


        filename_base = f"{name}"

        # print(name , 'type' , type(name))
        model =load_model(name)
        visualize_all(model ,x_test , y_test,
                      general_title=True ,
                      title = model_name.stem ,
                      save = True
                      )
        print('/-\\'*5, f'Model {name} visualization saved correctly','/-\\'*5)
    print('all models visualizations are saved correctly')



if __name__ == "__main__":
    # test these functions

    # print(load_split_train().head())
    # print(load_split_trainval().head())
    # print(load_split_val().head())
    # print(load_split_test().head())
    # print(load_kaggle().head())
    save_all_visualizations()
