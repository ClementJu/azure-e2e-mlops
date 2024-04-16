from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: Union[Path, str]) -> pd.DataFrame:
    if isinstance(path, str):
        path = Path(path)

    path = path.absolute()

    if not path.exists():
        raise RuntimeError(f'Cannot use non-existent path provided: {path}')

    csv_files = list(path.glob('*.csv'))

    if not csv_files:
        raise RuntimeError(f'No CSV files found in provided data path: {path}')

    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(df: pd.DataFrame) -> [np.array, np.array, np.array, np.array]:
    data_columns = df[[
        'Pregnancies',
        'PlasmaGlucose',
        'DiastolicBloodPressure',
        'TricepsThickness',
        'SerumInsulin',
        'BMI',
        'DiabetesPedigree',
        'Age']
    ].values

    label_column = df['Diabetic'].values

    return train_test_split(data_columns, label_column, test_size=0.30, random_state=0)
