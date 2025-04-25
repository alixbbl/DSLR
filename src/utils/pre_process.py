import pandas as pd
from datetime import datetime

def pre_process_df(data: pd.DataFrame, log_dir: str) -> pd.DataFrame:
    """
    Pre-process the input DataFrame by selecting numeric columns and turning categorical ones Birthdays and Best Hand as numeric ones.

    :param data: pd.DataFrame - The input DataFrame to be pre-processed.
    :return: pd.DataFrame - The pre-processed DataFrame.
    """
    pre_processed_df = data.copy()
    if 'Best Hand' in pre_processed_df.columns:
        pre_processed_df['Hand Binary'] = pre_processed_df['Best Hand'].map({'Left': 0, 'Right': 1})
    if 'Birthday' in pre_processed_df.columns:
        pre_processed_df['Birthday'] = pd.to_datetime(pre_processed_df['Birthday'])
        current_date = datetime.now()
        pre_processed_df['Age'] = (current_date - pre_processed_df['Birthday']).dt.days / 365.25
        pre_processed_df['Age'] = pre_processed_df['Age'].round().astype(int)
    #optional but useful 
    return pre_processed_df