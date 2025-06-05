import pandas as pd
from pathlib import Path
import argparse
from utils.upload_csv import upload_csv
from utils.maths import MyMaths
from utils.store import store_df_to_csv
from typing import Any

LOG_DIR = Path("output/describe")
FLOAT_FORMAT = 3

pd.options.display.float_format = lambda x: f"{x:.{FLOAT_FORMAT}f}"

def calculate_statistics(numeric_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate descriptive statistics for the numeric data.

    :param numeric_data: pd.DataFrame - The DataFrame containing only numeric columns.
    :return: pd.DataFrame - A DataFrame containing the descriptive statistics.
    """

    maths=MyMaths()
    row_fields=[
        'Count',
        'Mean',
        'Std',
        'Min',
        '25%',
        '50%',
        '75%',
        'Max',
        'Count NaN',
        'Percent NaN',
        'Variance',
        'Range',
        ]
    statistics_df = pd.DataFrame(index=row_fields, columns=numeric_data.columns)

    for column in numeric_data.columns:
        serie=numeric_data[column]
        statistics = [
            maths.my_count(serie),
            maths.my_mean(serie),
            maths.my_std(serie),
            maths.my_min(serie),
            maths.my_25percentile(serie),
            maths.my_median(serie),
            maths.my_75percentile(serie),
            maths.my_max(serie),
            maths.my_count_nan(serie),
            maths.my_percent_nan(serie),
            maths.my_var(serie),
            maths.my_range(serie)
        ]
        statistics_df[column]=statistics
    
    return statistics_df

def prepare_numeric_data(pre_processed_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the numeric data by selecting only numeric columns and dropping the 'Index' column if it exists.

    :param pre_processed_data: pd.DataFrame - The pre-processed DataFrame.
    :return: pd.DataFrame - The DataFrame containing only numeric columns.
    """
    
    numeric_data = pre_processed_data.select_dtypes(include="number")
    if 'Index' in numeric_data.columns:
        numeric_data = numeric_data.drop('Index', axis=1)
    if numeric_data.empty:
        raise ValueError("No numeric data found in the CSV file.")
    return numeric_data

# **************************** MAIN *******************************

def main(args):    
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    data=upload_csv(args.path_csv_to_read)
    
    numeric_data = prepare_numeric_data(data) 
    statistics = calculate_statistics(numeric_data)

    print(statistics)
    store_df_to_csv(statistics, "statistics", LOG_DIR, FLOAT_FORMAT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate descriptive statistics for a dataset")
    parser.add_argument('--path_csv_to_read',
                        type = str,
                        default='data/dataset_train.csv',
                        help="Path to the CSV file to analyze")
    parsed_args = parser.parse_args()
    main(parsed_args)