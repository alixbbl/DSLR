import pandas as pd

def store_df_to_csv(file_to_store:pd.DataFrame, filename: str, log_dir: str, float_format:int = 6) -> None:
    """
    Store the DataFrame to a CSV file.

    :param file_to_store: pd.DataFrame - The DataFrame to be stored.
    :param log_dir: str - The directory where the CSV file will be saved.

    """    
    float_format = f"%.{float_format}f"

    with open(f"{log_dir}/{filename}.csv", "w") as f:
        f.write(file_to_store.to_csv(index=True, float_format=float_format))