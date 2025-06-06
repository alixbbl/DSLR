import pandas as pd
import argparse
from utils.upload_csv import upload_csv
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

LOG_DIR = Path("output/pair_plots")


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data_num = data.select_dtypes(include=["float", "int"])
    if "Index" in data:
        data_num.drop("Index", axis=1, inplace=True)
    house_data = data["Hogwarts House"]
    data_num["Hogwarts House"] = house_data
    return data_num


def display_pair_plot(data_num: pd.DataFrame) -> None:
    colors = {
        "Gryffindor": "#8B0000",
        "Hufflepuff": "#B8860B",
        "Ravenclaw": "#00008B",
        "Slytherin": "#006400",
    }
    pair_plot = sns.pairplot(
        data_num,
        hue="Hogwarts House",
        vars=[
            "Arithmancy",
            "Astronomy",
            "Herbology",
            "Care of Magical Creatures",
            "Divination",
            "Defense Against the Dark Arts",
            "Transfiguration",
        ],
    )
    for ax in pair_plot.axes.flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel(ax.get_xlabel(), fontsize=6)
        ax.set_ylabel(ax.get_ylabel(), fontsize=6)

    pair_plot._legend.set_title("House")
    for text in pair_plot._legend.get_texts():
        text.set_fontsize(6)
    pair_plot._legend.get_title().set_fontsize(8)

    filename = LOG_DIR / "Hogwarts_pairplots.png"
    plt.savefig(filename)
    plt.show()


# **************************** MAIN *******************************


def main(parsed_args):

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    data = upload_csv(parsed_args.path_csv_to_read)
    if data is None:
        return
    data = clean_data(data)
    display_pair_plot(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_csv_to_read",
        type=str,
        help="CSV file path to read",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
