import pandas as pd
from pathlib import Path

def upload_csv(filepath: str | Path) -> pd.DataFrame:
    """
    This function loads a CSV file from the given path and returns a DataFrame.

    :param filepath: str - The path to the CSV file.
    :return: pd.DataFrame - The loaded DataFrame.
    """
    path = Path(filepath)
    if path.suffix.lower() != '.csv':
        raise ValueError("Invalid file format. Please provide a CSV file.")
    try:
        data = pd.read_csv(filepath, encoding='utf-8')
        if data.empty:
            raise ValueError("The CSV file is empty.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except pd.errors.ParserError:
        raise ValueError("Error parsing the CSV file. Check its format.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")
