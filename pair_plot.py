import pandas as pd
import argparse
from utils.upload_csv import upload_csv
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns

df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# Concaténation des DataFrames
df_concat = pd.concat([df1, df2], ignore_index=True)

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data_num = data.select_dtypes(include=['float', 'int'])
    data_num.drop('Index', axis=1, inplace=True) 
    # a ce stade on a perdu les maisons, il faut donc les remettre
    house_data = data['Hogwarts House']
    data_num['Hogwarts House'] = house_data
    return data_num

# Fignoler le resultat car pas de visibilite sur le Mac (trop faible) => readapter pour 42 pour corrections
def display_pair_plot(data_num: pd.DataFrame) -> None:
    
    colors = {'Gryffindor': '#8B0000', 'Hufflepuff': '#B8860B', 'Ravenclaw': '#00008B', 'Slytherin': '#006400'}
    pair_plot = sns.pairplot(data_num, hue='Hogwarts House', palette=colors, height=1.0, aspect=1.0)
    pair_plot.figure.set_size_inches(14, 10)
    plt.subplots_adjust(top=0.95, right=0.95, left=0.05, bottom=0.05)
    for ax in pair_plot.axes.flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel(ax.get_xlabel(), fontsize=6)
        ax.set_ylabel(ax.get_ylabel(), fontsize=6)

    pair_plot._legend.set_title('House')
    for text in pair_plot._legend.get_texts():
        text.set_fontsize(6)
    pair_plot._legend.get_title().set_fontsize(8)

    plt.show()


# Enlever une des variables correlees (Defense against) - a choisir l'une des var.


# **************************** MAIN *******************************

def main(parsed_args):
    data = upload_csv(parsed_args.path_csv_to_read)
    if data is None: 
        return
    if data['Hogwarts House'].dropna().empty:
        raise ValueError("Houses have not been assigned.")
    data = clean_data(data)
    display_pair_plot(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_csv_to_read',
                        nargs='?',
                        type=str,
                        help="CSV file path to read")
    parsed_args = parser.parse_args()
    main(parsed_args)