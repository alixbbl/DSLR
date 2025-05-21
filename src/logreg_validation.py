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
    
    validation_data_path_features = DATA_DIR / params.validation_data_path_features
    validation_data_path_target = DATA_DIR / params.validation_data_path_target
    
    if not validation_data_path_features.exists():
        print(f"Error : This file doesn't exist here -> {validation_data_path_features.resolve()}")
        return
    elif not validation_data_path_target.exists():
        print(f"Error : This file doesn't exist here -> {validation_data_path_target.resolve()}")
        return
    
    validation_dataset_features = upload_csv(str(validation_data_path_features))
    try:        
        print("Preparing test data...")
        params.standardize = False
        print(f"Loading model weights from {params.weights_file}...")
        predictor = LogisticRegressionPredictor()
        print("Making predictions...")
        predictions = predictor.predict(validation_dataset_features)
        save_predictions(predictions, LOG_DIR / "validation_predictions.csv") 

        # on verifie ensuite l'accuracy
        y_val = upload_csv(str(validation_data_path_target))
        accuracy = calculate_accuracy(predictions, y_val.values)
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        
    except Exception as e:
        print(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
