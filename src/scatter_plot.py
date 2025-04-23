import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from utils.upload_csv import upload_csv
from utils.constants import EXPECTED_LABELS_LIST

# Ce code doit permettre à l'utilisateur de visualiser si, par exemple, les élèves de Gryffondor 
# ont une certaine tendance à avoir de bonnes notes dans les deux matières ou s'il y a des 
# différences de performance entre les maisons.
# suite a l'histogramme on peut enlever deja : Care of Magical Creatures et Arithmancy

def display_correlation_matrix(data: pd.DataFrame)-> None:
    """
        Displays a correlation matrix to visualize correlation between courses.    
    """
    data_num = data.select_dtypes(include=['int', 'float'])
    data_num.drop('Index', axis=1, inplace=True) 
    corr_matrix = data_num.corr().abs()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

def display_scatters(houses: List[str], data: pd.DataFrame, courses_to_scatter: List[str]) -> None:
    """
        Display scatter plots for each house.
    """
    house_data_dict = {}
    df = data.select_dtypes(include=['int', 'float']).copy()
    df["Hogwarts House"] = data["Hogwarts House"]
    if courses_to_scatter == ['all']:
        course_list_new = ['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 
                           'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 
                           'Potions', 'Charms', 'Flying']
        for house in EXPECTED_LABELS_LIST:
            house_data_dict[house] = df[df['Hogwarts House'] == house].copy()  # On filtre les données de chaque maison
            if 'Index' in house_data_dict[house].columns:
                house_data_dict[house].drop('Index', axis=1, inplace=True)
            house_data_dict[house].drop('Hogwarts House', axis=1, inplace=True)
            
            for course in course_list_new:
                house_data_dict[house][course] = df.loc[df['Hogwarts House'] == house, course].values

        # nombre de combinaisons possibles
        n_courses = len(course_list_new)
        n_combinations = (n_courses * (n_courses - 1)) // 2  # nombre de combinaisons uniques

        # on peut ensute creer une grande figure avec plusieurs sous-graph
        rows = int(n_combinations**0.5) + 1
        cols = (n_combinations // rows) + 1
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        axes = axes.flatten()  # transforme en tableau 1D
        colors = {'Gryffindor': '#8B0000', 'Hufflepuff': '#B8860B', 'Ravenclaw': '#00008B', 'Slytherin': '#006400'}

        idx = 0
        for i, course1 in enumerate(course_list_new):
            for course2 in course_list_new[i+1:]:
                if idx < len(axes):
                    ax = axes[idx]
                    for house in houses:
                        ax.scatter(house_data_dict[house][course1], house_data_dict[house][course2],
                               label=house if idx == 0 else "", color=colors.get(house, '#000000'), alpha=0.5, s=8)
                    ax.set_xticklabels([])  # pour supprimer les étiquettes sur l'axe x
                    ax.set_yticklabels([])
                    ax.set_xlabel(course1, fontsize=8)
                    ax.set_ylabel(course2, fontsize=8)
                    ax.set_title(f'{course1} vs {course2}', fontsize=8)
                    idx += 1

        for j in range(idx, len(axes)):
            axes[j].axis('off')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, 
           loc='lower right',       # Position en bas à gauche
           fontsize='small',     # Texte plus petit
           title_fontsize='small', # Titre plus petit
           framealpha=0.7)         # Légère transparence du fond

        # partie mise en page
        plt.tight_layout()
        plt.subplots_adjust(right=0.9)  # on laisse un peu d'espace à gauche pour la légende
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.suptitle('Scatter Plots of All Course Combinations', fontsize=16)
        plt.show()

    elif len(courses_to_scatter) == 2:
        courses_dict = {
                    'astro': 'Astronomy',
                    'herbo': 'Herbology',
                    'def': 'Defense Against the Dark Arts',
                    'divin': 'Divination',
                    'mugg': 'Muggle Studies',
                    'runes': 'Ancient Runes',
                    'magic': 'History of Magic',
                    'trans': 'Transfiguration',
                    'potions': 'Potions',
                    'charms': 'Charms',
                    'flying': 'Flying'
        }
        course1 = courses_dict.get(courses_to_scatter[0])
        course2 = courses_dict.get(courses_to_scatter[1])
        colors = {'Gryffindor': '#FF0000', 'Hufflepuff': '#FFFF00', 'Ravenclaw': '#0000FF', 'Slytherin': '#00FF00'}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for house in EXPECTED_LABELS_LIST:
            house_data_dict[house] = df[df['Hogwarts House'] == house]
            ax.scatter(
                        house_data_dict[house][course1],
                        house_data_dict[house][course2],
                        label=house,
                        color=colors.get(house, '#000000'),
                        alpha=0.5
                    )
        ax.set_xlabel(course1)
        ax.set_ylabel(course2)
        ax.set_title(f'Scatter plot of "{course1}" vs "{course2}"', pad=20, fontweight='bold')
        ax.legend(title='House')
        fig.suptitle('Scatter Plot')
        plt.show()

    else:
        print("Invalid selection for courses. Please provide 'all' or two course names.")


# **************************** MAIN *******************************

def main(parsed_args):
    data = upload_csv(parsed_args.path_csv_to_read)
    if data is None: 
        return
    if 'Hogwarts House' not in data.columns:
        raise ValueError("The 'Hogwarts House' column is missing in the data.")
    # display_correlation_matrix(data)
    courses_to_scatter = (input("""Based on the correlation matrix, 
                                which courses do you want to display on a scatter plot 
                                (--choose 'all' or give two names in the following list :
                                    'astro' for Astronomy,
                                    'herbo' for Herbology,
                                    'def' for Defense Against the Dark Arts,
                                    'divin' for Divination,
                                    'mugg' for Muggle Studies,
                                    'runes' for Ancient Runes,
                                    'magic' for History of Magic,
                                    'trans' for Transfiguration,
                                    'potions' for Potions,
                                    'charms' for Charms,
                                    'flying' for Flying) : \n"""))
    courses_to_scatter_list = courses_to_scatter.split(" ")
    display_scatters(EXPECTED_LABELS_LIST, data, courses_to_scatter_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_csv_to_read',
                        type = str,
                        help = """CSV file to read""")
    parsed_args = parser.parse_args()
    main(parsed_args)