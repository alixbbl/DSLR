import pandas as pd
import argparse
from utils.upload_csv import upload_csv
import matplotlib.pyplot as plt
from typing import List

# Ce code doit permettre à l'utilisateur de visualiser si, par exemple, les élèves de Gryffondor 
# ont une certaine tendance à avoir de bonnes notes dans les deux matières ou s'il y a des 
# différences de performance entre les maisons.

def display_scatters(houses: List[str], data: pd.DataFrame, course1: str, course2: str) -> None:
    """
    Display scatter plots for each house.
    """
    courses_dict = {
                    'ari': 'Arithmancy',
                    'astro': 'Astronomy',
                    'herbo': 'Herbology',
                    'def': 'Defense Against the Dark Arts',
                    'divin': 'Divination',
                    'mugg': 'Muggle Studies',
                    'runes': 'Ancient Runes',
                    'magic': 'History of Magic',
                    'trans': 'Transfiguration',
                    'potions': 'Potions',
                    'care': 'Care of Magical Creatures',
                    'charms': 'Charms',
                    'flying': 'Flying'
    }
    colors = {'Gryffindor': '#FF0000', 'Hufflepuff': '#FFFF00', 'Ravenclaw': '#0000FF', 'Slytherin': '#00FF00'}
    fig, ax = plt.subplots(figsize=(10, 6))

    for house in houses:
        house_data = data[data['Hogwarts House'] == house]
        ax.scatter(house_data[courses_dict[course1]], house_data[courses_dict[course2]], label=house, color=colors.get(house, '#000000'), alpha=0.5)

    ax.set_xlabel(course1)
    ax.set_ylabel(course2)
    ax.set_title(f'Scatter plot of "{course1}" vs "{course2}"', pad=20, fontweight='bold')
    ax.legend(title='House')

    plt.show()

# **************************** MAIN *******************************

def main(parsed_args):

    if parsed_args.course1 == parsed_args.course2:
        raise ValueError("Course 2 must be different from Course 1.")
    data = upload_csv(parsed_args.path_csv_to_read)
    if data is None: 
        return
    if not data['Hogwarts House'].dropna().empty:
        houses = data['Hogwarts House'].unique()
    else:
        raise ValueError("Houses have not been assigned.")
    display_scatters(houses, data, parsed_args.course1, parsed_args.course2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    courses_list = [
        "ari", "astro", "herbo", "def", "divin", "mugg", "runes", "magic", 
        "trans", "potions", "care", "charms", "flying"]
    parser.add_argument('path_csv_to_read',
                        nargs='?',
                        type=str,
                        help="CSV file path to read")
    parser.add_argument('course1',
                        type=str,
                        choices=courses_list,
                        help="""Choose a course from the following list:  
                                ari: Arithmancy,
                                astro: Astronomy,
                                herbo: Herbology,
                                def: Defense Against the Dark Arts,
                                divin: Divination,
                                mugg: Muggle Studies,
                                runes: Ancient Runes,
                                magic: History of Magic,
                                trans: Transfiguration,
                                potions: Potions,
                                care: Care of Magical Creatures,
                                charms: Charms,
                                flying: Flying""")
    parser.add_argument('course2',
                        type=str,
                        choices=courses_list,
                        help="""Choose a course from the following list:  
                                ari: Arithmancy,
                                astro: Astronomy,
                                herbo: Herbology,
                                def: Defense Against the Dark Arts,
                                divin: Divination,
                                mugg: Muggle Studies,
                                runes: Ancient Runes,
                                magic: History of Magic,
                                trans: Transfiguration,
                                potions: Potions,
                                care: Care of Magical Creatures,
                                charms: Charms,
                                flying: Flying""")
    parsed_args = parser.parse_args()
    main(parsed_args)

# modifier pour pouvoir TOUT afficher, les 78 comibinaisons possibles 