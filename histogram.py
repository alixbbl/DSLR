import pandas as pd
import argparse
from utils.upload_csv import upload_csv
import matplotlib.pyplot as plt
import seaborn as sns

#  python -m histogram ./data/dataset_train.csv

# Ici melt() va simplifier et fondre le dataset pour autant d'entrees qu'il y a de scores 
# a un cours (un eleve sera donc 13 entrees au lieu d'une)
def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    """"
    This function converts the dataset to facilitate the plotting, each student becames an entry, a 
    column "Course" is created. Returns a new dataset.
    """
    columns_to_drop = ['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
    data_clean = data.drop(columns=columns_to_drop) # on supprime la data inutile
    data_melted = pd.melt(data_clean, id_vars=["Hogwarts House"], var_name="Course", value_name="Score") # Usage de melt()
    # print(data_melted.head())
    return data_melted

def display_histograms(data_melted: pd.DataFrame) -> None:
    
    sns.set_theme(style="whitegrid") # theme de seaborn pour láffichage
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 12)) # on divise l'espace en 16 volumes egaux
    fig.tight_layout(pad=3.0) # espace entre les graphes
    courses = data_melted['Course'].unique() #on recupere chaque nom de cours

    # on parcourt les cours, pour chacun on va creer un sous-graphe (subplot)
    for i, course in enumerate(courses):
        ax = axes[i // 4, i % 4] # posiitonner l'histo au bon endroit
        course_data = data_melted[data_melted['Course'] == course]
        # on peut ensuite creer l'histogramme pour chaque cours
        sns.histplot(course_data, x='Score', hue='Hogwarts House', multiple='stack', bins=20, palette='Set2', ax=ax)
        # Ajouter un titre pour chaque graphique
        ax.set_title(course, fontsize=10)
        ax.set_xlabel('Score', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        
    fig.suptitle("Score Distribution by Course and School", fontsize=12)
    plt.show()

# **************************** MAIN *******************************

def main(parsed_args):
    filepath = parsed_args.path_csv_to_read
    data = upload_csv(filepath)
    if data['Hogwarts House'].dropna().empty:
        raise ValueError("Houses have not been assigned.") # evite de travailler un dataset de type test
    data_melted = transform_data(data)
    display_histograms(data_melted)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('path_csv_to_read',
                        nargs='?',
                        type=str,
                        help="""CSV file to read""")
    parsed_args=parser.parse_args()
    main(parsed_args)