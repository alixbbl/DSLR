import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from utils.maths import MyMaths
from utils.upload_csv import upload_csv
from utils.store import store_df_to_csv

LOG_DIR = Path("output/histogram")

# ********************** SOLUTION 0 *******************************

def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    """"
    This function transforms the input DataFrame by dropping unnecessary columns 
    and melting the DataFrame to create a format suitable for plotting.

    :param data: pd.DataFrame - The dataset to transform.
    :return: pd.DataFrame - The transformed dataset.

    """
    columns_to_drop = ['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
    data_clean = data.drop(columns=columns_to_drop) 
    data_melted = pd.melt(data_clean, id_vars=["Hogwarts House"], var_name="Course", value_name="Score")
    print(data_melted.head())
    return data_melted

def display_histograms(data_melted: pd.DataFrame) -> None:

    sns.set_theme(style="whitegrid") # theme de seaborn pour láffichage
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 12)) # on divise l'espace en 16 volumes egaux
    fig.tight_layout(pad=3.0) # espace entre les graphes
    courses = data_melted['Course'].unique() #on recupere chaque nom de cours

    # on parcourt les cours, pour chacun on va creer un sous-graphe (subplot)
    for i, course in enumerate(courses):
        ax = axes[i // 4, i % 4] # positonner l'histo au bon endroit
        course_data = data_melted[data_melted['Course'] == course]
        # on peut ensuite creer l'histogramme pour chaque cours
        sns.histplot(course_data, x='Score', hue='Hogwarts House', multiple='stack', bins=20, palette='Set2', ax=ax)
        ax.set_title(course, fontsize=10)
        ax.set_xlabel('Score', fontsize=8)
        ax.set_ylabel('Nb students graded', fontsize=8)
        legend = ax.get_legend()
        if legend:
            legend.set_title("Poudlard Houses") 
            legend.get_title().set_fontsize(8)  
            for text in legend.get_texts():
                text.set_fontsize(7)
    total_plots = len(courses)
    for j in range(total_plots, 16):  # 16 = nrows * ncols
        fig.delaxes(axes[j // 4, j % 4]) 
 
    fig.suptitle("Score Distribution by Course and School", fontsize=12)
    plt.show()

# ********************** SOLUTION 1 *******************************

def display_histograms_without_melting(data: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 12))
    fig.tight_layout(pad=3.0)
        
    courses = [col for col in data.columns if col != 'Hogwarts House']
    
    # Plot each course
    for i, course in enumerate(courses):
        ax = axes[i // 4, i % 4]
        # Create separate histograms for each house
        for house in data['Hogwarts House'].unique():
            house_data = data[data['Hogwarts House'] == house]
            sns.histplot(house_data[course], bins=20, label=house, ax=ax)
            
        ax.set_title(course, fontsize=10)
        ax.set_xlabel('Score', fontsize=8)
        ax.set_ylabel('Nb students graded', fontsize=8)
        ax.legend(title="Poudlard Houses", fontsize=7, title_fontsize=8)
        
    total_plots = len(courses)
    for j in range(total_plots, 16):
        fig.delaxes(axes[j // 4, j % 4])
        
    fig.suptitle("Score Distribution by Course and School", fontsize=12)
    filename = LOG_DIR / "Hogwarts_histogram.png"
    plt.savefig(filename)
    plt.show()


# ********************** SOLUTION 2 *******************************

def identify_course_homogeneity(data: pd.DataFrame) -> dict:
    """
    Identify which course has the most homogeneous score distribution across houses.
    
    :param data: pd.DataFrame - The Hogwarts dataset
    :return: dict - Dictionary with course homogeneity metrics
    """
    maths = MyMaths()
    homogeneity_metrics = {}
    houses = data["Hogwarts House"].unique()
    courses = [col for col in data.columns if col != 'Hogwarts House']    
    
    for course in courses:   
        house_metrics = {}
        
        for house in houses:
            house_data = data[data["Hogwarts House"] == house][course].dropna()
                
            house_metrics[house] = {
                "mean": maths.my_mean(house_data),
                "std": maths.my_std(house_data),
                "count": maths.my_count(house_data)
            }
            
        house_means = [house_metrics[house]["mean"] for house in houses]
        between_variance = maths.my_var(house_means)

        within_variances = [house_metrics[house]["std"]**2 for house in houses]
        avg_within_variance = maths.my_mean(within_variances)
        
        f_ratio = between_variance / avg_within_variance if avg_within_variance > 0 else float('inf')
        
        homogeneity_metrics[course] = {
            "between_variance": between_variance,
            "avg_within_variance": avg_within_variance,
            "f_ratio": f_ratio,
            "house_metrics": house_metrics
        }

    return homogeneity_metrics

def plot_homogeneity_max_min(data: pd.DataFrame, metrics: dict, log_dir: Path):
    """
    Create histograms for courses showing distribution by house.
    
    Parameters:
    data (DataFrame): The Hogwarts dataset
    metrics (dict): The homogeneity metrics for each course
    output_dir (Path, optional): Directory to save plots
    """
    
    sorted_courses = sorted(metrics.items(), key=lambda x: x[1]["f_ratio"])
    
    most_homogeneous = sorted_courses[0][0]
    print(f"\nMost homogeneous course: {most_homogeneous} (F-ratio: {sorted_courses[0][1]['f_ratio']:.4f})")
    
    least_homogeneous = sorted_courses[-1][0]
    print(f"Least homogeneous course: {least_homogeneous} (F-ratio: {sorted_courses[-1][1]['f_ratio']:.4f})")
    
    plt.figure(figsize=(12, 10))
    
    # Plot for most homogeneous course
    plt.subplot(2, 1, 1)
    houses = data["Hogwarts House"].unique()
    for house in houses:
        house_data = data[data["Hogwarts House"] == house][most_homogeneous].dropna()
        if len(house_data) > 0:
            plt.hist(house_data, alpha=0.6, label=house, bins=10)
    
    plt.title(f"{most_homogeneous} - Most Homogeneous Course Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Plot for least homogeneous course
    plt.subplot(2, 1, 2)
    for house in houses:
        house_data = data[data["Hogwarts House"] == house][least_homogeneous].dropna()
        if len(house_data) > 0:
            plt.hist(house_data, alpha=0.6, label=house, bins=10)
    
    plt.title(f"{least_homogeneous} - Least Homogeneous Course Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def store_homogeneity_metrics(metrics: dict, log_dir: Path):
    """Store homogeneity metrics in a CSV file.

    :param metrics: dict - Dictionary with course homogeneity metrics
    :param log_dir: Path - Directory to save the CSV file
    :return: None
    """
    records = []
    for course, data in metrics.items():
        record = {
            'Course': course,
            'Between_Variance': data['between_variance'],
            'Avg_Within_Variance': data['avg_within_variance'],
            'F_Ratio': data['f_ratio']
        }
        records.append(record)
    
    metrics_df = pd.DataFrame(records)
    metrics_df = metrics_df.sort_values('F_Ratio')

    store_df_to_csv(metrics_df, "course_homogeneity_metrics", LOG_DIR, 2)
    print(f"Homogeneity metrics saved to {LOG_DIR}/course_homogeneity_metrics.csv")

# **************************** MAIN *******************************

def main(parsed_args):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    filepath = parsed_args.path_csv_to_read
    data = upload_csv(filepath)

    columns_to_drop = ['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
    data = data.drop(columns=columns_to_drop)
    
    #Solution 0
    # data_melted = transform_data(data)
    # store_df_to_csv(data_melted, "dataset_histogram", LOG_DIR, 2)
    # display_histograms(data_melted)
    
    #Solution 1
    display_histograms_without_melting(data)

    #Solution 2
    # metrics = identify_course_homogeneity(data)
    # plot_homogeneity_max_min(data, metrics, LOG_DIR)
    # store_homogeneity_metrics(metrics, LOG_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_csv_to_read',
                        type = str,
                        default='data/dataset_train.csv',
                        help = """CSV file to read""")
    parsed_args = parser.parse_args()
    main(parsed_args)