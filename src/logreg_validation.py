import pandas as pd
import numpy as np
from pathlib import Path
from utils.config import params
from utils.upload_csv import upload_csv
from utils.maths import MyMaths
from utils.metrics import calculate_accuracy
from utils.store import save_predictions
from logreg_predict import LogisticRegressionPredictor

LOG_DIR = params.LOG_DIR
DATA_DIR = params.DATA_DIR
HOGWART_HOUSES = params.hogwart_houses
TRAINING_FEATURES = params.training_features


# ****************************************** MAIN ***************************************************

def main():

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    validation_data_path = DATA_DIR / params.validation_data_path
    if not validation_data_path.exists():
        print(f"Error : This file doesn't exist here -> {validation_data_path.resolve()}")
        return
    
    validation_dataset = upload_csv(str(validation_data_path))
    try:        
        print("Preparing test data...")
        X_val = validation_dataset[TRAINING_FEATURES].copy() # X_val ici est deja standardized et les NaN sont absents
        y_val = validation_dataset['Hogwarts House']
        params.standardize = False
        print(f"Loading model weights from {params.weights_file}...")
        predictor = LogisticRegressionPredictor()
        print("Making predictions...")
        predictions = predictor.predict(X_val)
        save_predictions(predictions, LOG_DIR / "validation_predictions.csv") 

        # on verifie ensuite l'accuracy 
        accuracy = calculate_accuracy(predictions, y_val.values)
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        
    except Exception as e:
        print(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
