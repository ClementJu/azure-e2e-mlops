from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def train_model(reg_rate: float, X_train: np.array, y_train: np.array) -> LogisticRegression:
    return LogisticRegression(C=1 / reg_rate, solver='liblinear').fit(X_train, y_train)


def score_model(trained_model: LogisticRegression, X_test: np.array, y_test: np.array) -> float:
    accuracy = trained_model.score(X_test, y_test)
    return accuracy


def export_model(trained_model: LogisticRegression, export_directory_path: Path = Path('.'),
                 model_name: str = 'sklearn_regression_model.joblib') -> None:
    export_directory_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(value=trained_model, filename=export_directory_path / model_name)
