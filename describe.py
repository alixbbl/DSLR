import pandas as pd
import argparse
from utils.upload_csv import upload_csv
from utils.maths import MyMaths
from typing import Any

pd.options.display.float_format = '{:.6f}'.format # affiche les 6 decimales, comportement par defaut de describe()

def do_the_maths(numeric_data: pd.DataFrame) -> pd.DataFrame:

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
        ]
    col_fields = [name for name in numeric_data.columns if name != "Index"]
    describe_df = pd.DataFrame(index=row_fields, columns=col_fields)

    for feature in col_fields:
        serie=numeric_data[feature]
        count=maths.my_count(serie)
        mean=maths.my_mean(serie)
        std=maths.my_std(serie)
        min=maths.my_min(serie)
        perc25=maths.my_25percentile(serie)
        perc50=maths.my_median(serie)
        perc75=maths.my_75percentile(serie)
        max=maths.my_max(serie)

        row=[count, mean, std, min, perc25, perc50, perc75, max]
        describe_df[feature]=row
    
    return describe_df

# **************************** MAIN *******************************

def main(parsed_args):
    filepath=parsed_args.path_csv_to_read
    data=upload_csv(filepath)
    # print(data.dtypes)
    numeric_data = data.select_dtypes(include=['float','int'])
    if 'Hogwarts House' in numeric_data.columns:
        numeric_data = numeric_data.drop('Hogwarts House', axis=1)
    # print(numeric_data.shape)
    to_display = do_the_maths(numeric_data)
    print(to_display)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('path_csv_to_read',
                        nargs='?',
                        type=str,
                        help="""CSV file to read""")
    parsed_args=parser.parse_args()
    main(parsed_args)