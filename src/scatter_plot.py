import pandas as pd
import numpy as np
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from utils.store import store_df_to_csv
from utils.upload_csv import upload_csv

# Ce code doit permettre à l'utilisateur de visualiser si, par exemple, les élèves de Gryffondor 
# ont une certaine tendance à avoir de bonnes notes dans les deux matières ou s'il y a des 
# différences de performance entre les maisons.
# suite a l'histogramme on peut enlever deja : Care of Magical Creatures et Arithmancy

LOG_DIR = Path("output/scatter_plots")

def display_correlation_matrix(numeric_data: pd.DataFrame)-> pd.DataFrame:
    """
        Displays a correlation matrix to visualize correlation between courses.    
        :param data: DataFrame containing the data.
        :return: pd.DataFrame with the correlation matrix.
    """
    corr_matrix = numeric_data.corr(method="pearson")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar=False)
    plt.title('Correlation Matrix')
    plt.show()
    return corr_matrix

def scatter_plot_correlation_matrix(correlation_df: pd.DataFrame, threshold: float, data: pd.DataFrame) -> None:
    """
    Displays scatter plots for all course pairs with correlation above threshold.
    
    :param correlation_df: DataFrame containing the correlation matrix
    :param threshold: Threshold for displaying correlations
    :param data: Original DataFrame with all the data (including Hogwarts House)
    :return: None
    """
    courses = correlation_df.columns
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
                plt.show()
                plt.savefig(f'{LOG_DIR}/{course1}_vs_{course2}.png')
                plt.close()

# **************************** MAIN *******************************

def main(parsed_args):
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    data = upload_csv(parsed_args.path_csv_to_read)
    
    numeric_data = data.select_dtypes(include="number")
    numeric_data = numeric_data.drop(columns=numeric_data.columns[0], axis=1)
    data = data.drop(columns=data.columns[0], axis=1)

    correlation_matrix = display_correlation_matrix(numeric_data)
    store_df_to_csv(correlation_matrix, "correlation_matrix", LOG_DIR, 2)

    print(data.head())
    scatter_plot_correlation_matrix(correlation_matrix, 0.70, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_csv_to_read',
                        type = str,
                        help = """CSV file to read""")
    parsed_args = parser.parse_args()
    main(parsed_args)