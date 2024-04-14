from pathlib import Path
import pytest

from data import load_data, split_data


def test_load_data_should_work_with_path(dataset_directory: Path):
    result = load_data(dataset_directory)
    assert len(result) == 10


def test_load_data_should_work_with_string(dataset_directory: Path):
    result = load_data(str(dataset_directory))
    assert len(result) == 10


def test_load_data_should_raise_when_no_files_are_found():
    with pytest.raises(RuntimeError) as error:
        load_data('./')
    assert error.match('No CSV files found in provided data')


def test_load_data_should_raise_when_path_does_not_exist():
    with pytest.raises(RuntimeError) as error:
        load_data('/invalid/path/does/not/exist/')
    assert error.match('Cannot use non-existent path provided')


def test_split_data_should_work(dataset_directory: Path):
    dataset_df = load_data(path=dataset_directory)
    X_train, X_test, y_train, y_test = split_data(df=dataset_df)
    assert len(X_train) == 7
    assert len(y_train) == 7
    assert len(X_test) == 3
    assert len(y_test) == 3
