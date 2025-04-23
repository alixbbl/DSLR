import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from utils.upload_csv import upload_csv
from utils.constants import EXPECTED_LABELS_LIST

LOG_DIR = "output/scatter_plots"

def get_numeric_data(data: pd.DataFrame) -> pd.DataFrame:
    """
        Get numeric data from the DataFrame.

        :param data: DataFrame containing all the data.
        :return: DataFrame with only the numeric data left.
    """
    data.dropna(inplace=True)
    numeric_data = data.select_dtypes(include=['int', 'float'])
    if 'Index' in numeric_data.columns:
        numeric_data = numeric_data.drop('Index', axis=1)
    if numeric_data.empty:
        raise ValueError("No numeric data found in the CSV file.")
    return numeric_data


def store_correlation(correlation_df: pd.DataFrame) -> None:
    """
        Store the correlation matrix in a CSV file.

        :param correlation_df: DataFrame containing the correlation matrix.
        :return: None
    """
    rounded_corr = correlation_df.round(2)
    corrolation = tabulate(rounded_corr, headers='keys', tablefmt='pretty')
    with open(f'{LOG_DIR}/correlation.csv', 'w') as f:
        f.write(corrolation)

def get_correlation(numeric_data: pd.DataFrame)-> None:
    """
        Displays a correlation matrix as a simple table to visualize
        correlation between courses.

        :param data: DataFrame containing the data.
        :return: None
    """
    correlation_df = numeric_data.corr()
    store_correlation(correlation_df)
    return correlation_df


def scatter_plot_correlation(correlation_df: pd.DataFrame, threshold: float, data: pd.DataFrame) -> None:
    """
    Displays scatter plots for all course pairs with correlation above threshold.
    
    :param correlation_df: DataFrame containing the correlation matrix
    :param threshold: Threshold for displaying correlations
    :param data: Original DataFrame with all the data (including Hogwarts House)
    :return: None
    """
    courses = correlation_df.columns
    plt.figure(figsize=(10, 8))
    plot_count = 0
    
    # Iterate through pairs of courses
    for i, course1 in enumerate(courses):
        for j, course2 in enumerate(courses):
            # skip self-comparisons
            if i >= j:
                continue
                
            # Get absolute correlation value (no negative)
            corr_value = abs(correlation_df.loc[course1, course2])
            
            if corr_value >= threshold:
                plot_count += 1
                
                # Create a new figure for each plot
                plt.figure(figsize=(10, 7))
                
                scatter = plt.scatter(
                    data[course1], 
                    data[course2],
                    c=pd.factorize(data['Hogwarts House'])[0],
                    alpha=0.7,
                    cmap='viridis'
                )
                
                # Add trend line
                z = np.polyfit(data[course1], data[course2], 1)
                p = np.poly1d(z)
                x_range = np.linspace(data[course1].min(), data[course1].max(), 100)
                plt.plot(x_range, p(x_range), "r--", alpha=0.8)
                
                # Calculate actual correlation (not absolute value)
                actual_corr = correlation_df.loc[course1, course2]
                
                # Format titles and labels
                plt.title(f'Correlation: {course1} vs {course2}\nr = {actual_corr:.2f}', fontsize=14)
                plt.xlabel(course1, fontsize=12)
                plt.ylabel(course2, fontsize=12)
                
                # legend for houses
                houses = data['Hogwarts House'].unique()
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                  markersize=10, label=house) 
                                  for i, house in enumerate(houses)]
                plt.legend(handles=legend_elements, title="Hogwarts House")
                
                # for better readability
                plt.grid(True, alpha=0.3)
                
                if actual_corr > 0:
                    correlation_type = "Positive"
                else:
                    correlation_type = "Negative"
                    
                plt.annotate(
                    f"{correlation_type} Correlation: {abs(actual_corr):.2f}",
                    xy=(0.05, 0.95),
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
                
                plt.tight_layout()
                plt.savefig(f'{LOG_DIR}/{course1}_vs_{course2}.png')
                plt.close()

# **************************** MAIN *******************************

def main(parsed_args):
    if Path(LOG_DIR).exists() == False:
        Path((LOG_DIR)).mkdir(parents=True, exist_ok=True)
    data = upload_csv(parsed_args.path_csv_to_read)
    numeric_data = get_numeric_data(data)
    correlation_df = get_correlation(numeric_data)
    scatter_plot_correlation(correlation_df, 0.70, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_csv_to_read',
                        type = str,
                        help = """CSV file to read""")
    parsed_args = parser.parse_args()
    main(parsed_args)