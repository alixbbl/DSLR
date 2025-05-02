import pandas as pd

def upload_csv(filepath: str) -> pd.DataFrame:
    """
    This function loads a CSV file from the given path and returns a DataFrame.

    :param filepath: str - The path to the CSV file.
    :return: pd.DataFrame - The loaded DataFrame.
    """
    if not filepath.endswith('.csv'):
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
