import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def cleanup(export_directory_path: Path):
    yield
    shutil.rmtree(export_directory_path, ignore_errors=True)


@pytest.fixture
def dataset_directory() -> Path:
    current_directory = Path(__file__).parent.absolute()
    return current_directory / 'datasets'


@pytest.fixture
def export_directory_path() -> Path:
    return Path('./to_delete')


