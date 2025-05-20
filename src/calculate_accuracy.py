import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from utils.upload_csv import upload_csv


def calculate_accuracy(y_pred, y_true) -> float:
    """
    Calculate the accuracy of the model on test data

    :param y_pred: Calculated predictions
    :param y_true: True labels (Hogwarts Houses)
    :return: accuracy score (0-1)
    """
    correct = sum(1 for pred, true in zip(y_pred, y_true) if pred == true)
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0
    return accuracy


# ************************************* MAIN ******************************************

def main(parsed_args):
    data_truth = upload_csv(parsed_args.path_truth_csv)

    if 'Index' in data_truth.columns:
        data_truth = data_truth.drop(columns=['Index'])
    data_houses = upload_csv(parsed_args.path_houses_csv)

    if 'Index' in data_houses.columns:
        data_houses = data_houses.drop(columns=['Index'])

    y_pred = data_houses['Hogwarts House']
    y_true = data_truth['Hogwarts House']
    accuracy = calculate_accuracy(y_pred, y_true)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_houses_csv',
                        type=str,
                        help='CSV file with predicted houses (houses.csv)')
    parser.add_argument('path_truth_csv',
                        type=str,
                        help='CSV file with true values (ground truth)')
    parsed_args = parser.parse_args()
    main(parsed_args)
