import csv
import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt

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


def write_output_thetas(list_thetas: List[Dict]) -> None:
    """"
        This functions returns a file with the thetas for predictions.
    """
    output_file = "thetas.csv"
    print(list_thetas)
    with open(output_file, mode='w', newline = '') as file:
        writer = csv.writer(file)
        courses = ['Ancient Runes', 'Astronomy', 'Charms', 'Defense Against the Dark Arts', 'Divination', 'Herbology', 'Flying']
        header = ['Hogwarts House'] + courses
        writer.writerow(header)

        for house, thetas in list_thetas.items():
            writer.writerow([house] + thetas.squeeze().tolist())
 
    print(f"{output_file} is printed ✅ !")


def write_output_predictions(list_predictions: List[str]) -> None:
    """"
        This functions returns a file with the predictions.
    """
    output_file = "predictions.csv"
    with open(output_file, mode='w', newline = '') as file:
        writer = csv.writer(file)
        header = ['Hogwarts House']
        writer.writerow(header)
        for house in list_predictions:
            writer.writerow([house])
 
    print(f"{output_file} is printed ✅ !")


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
