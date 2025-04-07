import csv
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from .constants import TRAINING_FEATURES_LIST

def log_loss(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Computes the binary cross-entropy (log loss) for logistic regression.
    Args:
        y_true (np.array): Array of true labels (0 or 1).
        y_pred (np.array): Array of predicted probabilities (between 0 and 1).
    Returns the average log loss (float).
    """
    epsilon = 1e-15  # To avoid log(0), which is undefined
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Keeps values in the range (0,1)
    loss_value = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss_value

def write_output_constants_standard(mean_const: pd.Series, std_const: pd.Series) -> None:
    """"
        This functions returns a file with the datatrain mean and std for each feature,
        to standardize the dataset_test.
    """
    output_file = "standardization_constants.csv"
    with open(output_file, mode='w', newline = '') as file:
        writer = csv.writer(file)
        header = ['Feature', 'Mean', 'Std']
        writer.writerow(header)
        for feature in mean_const.index:
            writer.writerow([feature, mean_const[feature], std_const[feature]])
    print(f"{output_file} is printed !")

def write_output_thetas(list_thetas: List[Dict]) -> None:
    """"
        This functions returns a file with the thetas for predictions.
    """
    output_file = "thetas.csv"
    with open(output_file, mode='w', newline = '') as file:
        writer = csv.writer(file)
        courses = TRAINING_FEATURES_LIST
        header = ['Hogwarts House'] + courses
        writer.writerow(header)
        for house, thetas in list_thetas.items():
            writer.writerow([house] + thetas.squeeze().tolist())
    print(f"{output_file} is printed !")

def write_output_predictions(list_predictions: List[str]) -> None:
    """
    This function writes the predictions to a CSV file.
    """
    output_file = "houses.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Index', 'Hogwarts House']
        writer.writerow(header)
        for i, house in enumerate(list_predictions, start=0):
            writer.writerow([i, house])
    print(f"{output_file} is printed!")

def plot_cost_report(list_cost_report):
    """
        This functions print a visualization of the log loss report.
    """
    plt.plot(list_cost_report)
    plt.title("Evolution de la fonction de coût (Log Loss)")
    plt.xlabel("Iterations")
    plt.ylabel("Coût")
    plt.grid(True)
    plt.show()

