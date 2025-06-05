import pandas as pd

def store_df_to_csv(file_to_store:pd.DataFrame, filename: str, log_dir: str, float_format:int = 6) -> None:
    """
    Store the DataFrame to a CSV file.

    :param file_to_store: pd.DataFrame - The DataFrame to be stored.
    :param log_dir: str - The directory where the CSV file will be saved.

    """    
    float_format = f"%.{float_format}f"

    with open(f"{log_dir}/{filename}.csv", "w") as f:
        f.write(file_to_store.to_csv(index=False, float_format=float_format))


def save_predictions(predictions, output_file):
    """
        Save predictions to a CSV file
        
        :param predictions: list of predicted houses
        :param output_file: path to output CSV file
    """
    output = pd.DataFrame({'Hogwarts House': predictions})
    output.to_csv(output_file, index_label='Index')
    print(f"Predictions saved to {output_file}")