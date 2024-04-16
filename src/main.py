import argparse
from pathlib import Path

import mlflow
from data import load_data, split_data
from train import train_model, export_model, score_model


def main(args):
    mlflow.autolog()

    with mlflow.start_run():
        dataset_df = load_data(path=args.training_data)
        X_train, X_test, y_train, y_test = split_data(df=dataset_df)
        trained_model = train_model(reg_rate=args.reg_rate, X_train=X_train, y_train=y_train)

        test_accuracy = score_model(trained_model=trained_model, X_test=X_test, y_test=y_test)
        mlflow.log_metric('test_accuracy', test_accuracy)

        export_model(trained_model=trained_model, export_directory_path=Path('./results'))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)

    return parser.parse_args()


if __name__ == "__main__":
    print("\n\n")
    print("*" * 60)

    parsed_arguments = parse_args()
    main(parsed_arguments)

    print("*" * 60)
    print("\n\n")
