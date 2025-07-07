'''
    this file wll contain the main logic of
    - training the model , do some printing also
    - evaluate the modle
    - and saving the modle
'''
import numpy as np
from sklearn import linear_model
from sklearn import metrics
import pickle
from datetime import datetime
import os


def _confirm_dtype(x, y, dtype=np.ndarray):
    if not isinstance(x, dtype) or not isinstance(y, dtype):
        return False
    return True


def _confirm_outer_shape(x, y):
    if x.shape[0] != y.shape[0]:
        return False
    return True


def _check_and_raise(x, y):
    # check if they are of np.array type
    if not _confirm_dtype(x, y, dtype=np.ndarray):
        raise ValueError("X and must be of type np.array")

    # check rows of x match y
    if not _confirm_outer_shape(x, y):
        raise ValueError(" rows or x , and y  are not the same")


def save_model(model, save_dir="models"):
    """
    Saves a model to a .pkl file with a timestamp and model type in its name.

    Args:
        model: The model object to save.
        save_dir (str): The directory where the model should be saved.
                        Defaults to "models".
    """
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # Get the model's class name
    model_name = type(model).__name__

    # Create a safe timestamp string for filenames
    # Replace slashes, colons, and spaces with underscores or hyphens
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Construct the filename
    filename = f"{model_name}_{timestamp}.pkl"
    filepath = os.path.join(save_dir, filename)

    try:
        with open(filepath, "wb") as file:
            pickle.dump(model, file)
        print(f"Model saved successfully to: {filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")


def train(model: linear_model, x: np.array, y: np.array):

    _check_and_raise(x, y)

    print("=====================Fitting a linear Model =================")

    model.fit(x, y)

    print("=====================Finished================================")


def evaluate(model: linear_model, x: np.array, y_true: np.array):
    _check_and_raise(x, y_true)

    rmse_error = metrics.root_mean_squared_error(y_true=y_true, y_pred=model.predict(x))
    r2_error = metrics.r2_score(y_true=y_true, y_pred=model.predict(x))

    print(f"{rmse_error=} , {r2_error=}")

    return {"rms_error": rmse_error, "r2_score": r2_error}
