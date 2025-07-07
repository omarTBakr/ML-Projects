'''
this module contains the all custom classed that i have  used like

 - custom logestic regression (using torch ) with focal losss
 - custom voting classifier since sklearn's voting requires fitting and i have already saved models
 - NN class with focal loss

 Note: there are a big chunk of duplication in this module specifically
 with LogesticRegressionWithFocalLoss class and
     NNWithFocalLoss

we could have made the cutom model that inheritis from sklear generic and
give it whatever model we want like logesticRegression or NN
but i avoided doing so because there are dpenndices built over that structure
so i can not change it
'''



import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin
import  torch
import  pandas as pd
import  numpy as np
from torchvision.ops import sigmoid_focal_loss
from torch import  nn








##############################voting clf#################################3
class CustomVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    A custom Voting Classifier that works with pre-fitted models.
    It handles inconsistent output shapes from different estimators.
    """

    def __init__(self, estimators, voting="soft", weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights

    def fit(self, X, y=None):
        """
        The fit method is a placeholder. It validates the estimators and sets the
        'classes_' attribute. It does NOT re-train the models.
        """
        model_list = [model for name, model in self.estimators]
        if not model_list:
            raise ValueError("Estimators list cannot be empty.")

        self.classes_ = getattr(model_list[0], "classes_", None)
        if self.classes_ is None:
            # Fallback for models that might not have the attribute post-loading
            # This is a robust way to handle different types of saved models
            print("Warning: classes_ not found on first estimator. Inferring from unique values in y.")
            self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """
        Generates predictions for X based on the voting rule.
        """

        predictions_list = [np.squeeze(model.predict(X)) for name, model in self.estimators]

        # print("prediction list  shapes ", [pred.shape for pred in predictions_list])
        predictions_array = np.stack(predictions_list, axis=1)

        if self.voting == "hard":
            # Perform majority vote for each sample
            majority_vote = mode(predictions_array, axis=1, keepdims=False)[0]
            return majority_vote
        else:  # soft voting
            avg_probas = self.predict_proba(X)
            return np.argmax(avg_probas, axis=1)



    def _predict_proba_soft(self, X):
        probas_list = []

        for name, model in self.estimators:
            probas = model.predict_proba(X)
            probas = np.squeeze(probas)
            # print("probability shape  for ", name, "is", probas.shape)

            probas_list.append(probas)

        # Now all probas are (n_samples, n_classes)
        probas_stacked = np.stack(probas_list, axis=0)  # shape (n_estimators, n_samples, n_classes)
        avg_probas = np.average(probas_stacked, axis=0, weights=self.weights)  # shape (n_samples, n_classes)
        return avg_probas

    def predict_proba(self, X):
        """
        Computes weighted average probabilities for each class across all estimators.
        Ensures all probability arrays are of shape (n_samples, n_classes).
        """

        return self._predict_proba_soft(X)

#####################################logestic regression with focal loss ###############################333333
device = "cuda" if torch.cuda.is_available() else "cpu"  # ther is a problem with cuda in my device




class LogisticRegressionTorch(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        # We return the raw logits. The loss function will handle the sigmoid.
        return self.linear(x)




class LogisticRegressionWithFocalLoss(BaseEstimator, ClassifierMixin):

    def __init__(self, n_input_features=None, alpha=0.25, gamma=2.0, lr=0.01, n_epochs=100, weight_decay=0):

        self.n_input_features = n_input_features
        self.alpha = alpha
        self.gamma = gamma
        self.lr = lr
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay  # For L2 Regularization

        self.model = None
        self.classes_ = torch.tensor([0, 1])  # Sklearn expects this attribute

    def _convert_to_tensor(self, x):

        if not isinstance(x, torch.Tensor):
            x = pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x
            x = torch.as_tensor(x.values, dtype=torch.float32, device=device)
        return x

    def fit(self, X, y):
        # FIX: Infer input features from the data in fit()
        # This is more robust and required by scikit-learn
        if self.model is None:
            self.n_input_features = X.shape[1]
            self.model = LogisticRegressionTorch(self.n_input_features).to(device)

        X, y = self._convert_to_tensor(X), self._convert_to_tensor(y)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        batch_size = 512
        batch_start = torch.arange(0, len(X), batch_size)

        for epoch in range(self.n_epochs):
            self.model.train()
            for start in batch_start:
                X_batch = X[start : start + batch_size]
                y_batch = y[start : start + batch_size]

                y_pred_logits = self.model(X_batch)

                loss = sigmoid_focal_loss(y_pred_logits, y_batch, alpha=self.alpha, gamma=self.gamma, reduction="mean")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # This attribute signals to sklearn that the model has been fitted
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = self._convert_to_tensor(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            # Apply sigmoid since ( as the model only return logits)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
        return preds.cpu().numpy()

    def predict_proba(self, X):
        X = self._convert_to_tensor(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.sigmoid(logits).unsqueeze(-1)
            # Return shape (n_samples, 2) with [prob_of_0, prob_of_1] just like sklearn
            return torch.cat([1 - probs, probs], dim=1).cpu().numpy()

class NNTorch(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.model =nn.Sequential (
            #######first layer ###########
            nn.Linear(n_input_features , 32) ,
            nn.BatchNorm1d(32) ,
            nn.ReLU() ,
            nn.Dropout() ,
            ###########second layer##########3
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(),
            ###########thrid layer##########
            nn.Linear(16, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(),

            #################fourth##########
            nn.Linear(16, 2),
            nn.BatchNorm1d(4),
            nn.ReLU(),

            #############fith layer#######
            nn.Linear(4, 1),

        )

    def forward(self, x):
        # We return the raw logits. The loss function will handle the sigmoid.
        return self.model(x)



class NNWithFocalLoss(BaseEstimator, ClassifierMixin):

    def __init__(self, n_input_features=None, alpha=0.25, gamma=2.0, lr=0.01, n_epochs=100, weight_decay=0):

        self.n_input_features = n_input_features
        self.alpha = alpha
        self.gamma = gamma
        self.lr = lr
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay  # For L2 Regularization

        self.model = None
        self.classes_ = torch.tensor([0, 1])  # Sklearn expects this attribute

    def _convert_to_tensor(self, x):

        if not isinstance(x, torch.Tensor):
            x = pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x
            x = torch.as_tensor(x.values, dtype=torch.float32, device=device)
        return x

    def fit(self, X, y):
        # FIX: Infer input features from the data in fit()
        # This is more robust and required by scikit-learn
        if self.model is None:
            self.n_input_features = X.shape[1]
            self.model = LogisticRegressionTorch(self.n_input_features).to(device)

        X, y = self._convert_to_tensor(X), self._convert_to_tensor(y)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        batch_size = 512
        batch_start = torch.arange(0, len(X), batch_size)

        for epoch in range(self.n_epochs):
            self.model.train()
            for start in batch_start:
                X_batch = X[start : start + batch_size]
                y_batch = y[start : start + batch_size]

                y_pred_logits = self.model(X_batch)

                loss = sigmoid_focal_loss(y_pred_logits, y_batch, alpha=self.alpha, gamma=self.gamma, reduction="mean")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # This attribute signals to sklearn that the model has been fitted
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = self._convert_to_tensor(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            # Apply sigmoid since ( as the model only return logits)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
        return preds.cpu().numpy()

    def predict_proba(self, X):
        X = self._convert_to_tensor(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.sigmoid(logits).unsqueeze(-1)
            # Return shape (n_samples, 2) with [prob_of_0, prob_of_1] just like sklearn
            return torch.cat([1 - probs, probs], dim=1).cpu().numpy()

