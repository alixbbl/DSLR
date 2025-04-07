import pandas as pd

# la fonction read_csv prend en charge la verification du path et procede a la mise en flux comme open()
# on peut donc se focus sur les erreurs systeme et le parsing, en utilisant directement pd.read_csv() et ses options
def upload_csv(filepath: str) -> pd.DataFrame:
    """
    This function loads a CSV file from a given path and returns a DataFrame.
    """
    if not filepath.endswith('.csv'):
        raise ValueError("Invalid file format. Please provide a CSV file.")

    try:
        data = pd.read_csv(filepath, encoding='utf-8')
        if data.empty:
            raise ValueError("The CSV file is empty.")
        # print(f"Successfully loaded a CSV file of dimensions: {data.shape}!")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty or unreadable.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing the CSV file. Check its format.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")
