from pathlib import Path
from sklearn.linear_model import LogisticRegression

from data import load_data, split_data
from train import train_model, export_model


def test_train_model_should_return_trained_model(dataset_directory: Path):
    dataset_df = load_data(path=dataset_directory)
    X_train, _, y_train, _ = split_data(df=dataset_df)
    trained_model = train_model(reg_rate=0.1, X_train=X_train, y_train=y_train)
    assert isinstance(trained_model, LogisticRegression)
    assert trained_model.C == 1 / 0.1


def test_export_model_should_work(export_directory_path: Path):
    model = LogisticRegression()
    model_name = 'my_test_model.joblib'

    export_model(trained_model=model, export_directory_path=export_directory_path, model_name=model_name)
    assert export_directory_path.exists()
    files_in_directory = list(export_directory_path.glob('*'))
    assert len(files_in_directory) == 1
    assert str(files_in_directory[0]).endswith(model_name)
