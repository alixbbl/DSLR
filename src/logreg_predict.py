import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from utils.config import params
from utils.upload_csv import upload_csv
from utils.metrics import calculate_accuracy
from utils.store import save_predictions

LOG_DIR = params.LOG_DIR
DATA_DIR = params.DATA_DIR
HOGWART_HOUSES = params.hogwart_houses
TRAINING_FEATURES = params.training_features

class DataPreProcessor:
    def __init__(self, data_path: str):
        self.data = upload_csv(data_path)

    def ft_imputation_by_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Replaces all the NaN by the mean of each Series."""
        prediction_dataset = pd.DataFrame()
        for feature in df.columns:
            prediction_dataset[feature] = df[feature]
        prediction_dataset.fillna(prediction_dataset.mean(), inplace=True) # imputation par la moyenne
        return prediction_dataset

    def pre_process(self):
        self.X = self.data[TRAINING_FEATURES].copy()
        return self.ft_imputation_by_mean(self.X) 
    
    def pre_process_truth(self, dataset_path):
        data_truth = upload_csv(dataset_path)
        if 'Index' in data_truth.columns:
            data_truth.drop(columns='Index', inplace=True)
        return data_truth['Hogwarts House']

class LogisticRegressionPredictor:
    def __init__(self, weights_file=None):
        self.weights_file = params.weights_file if weights_file else str(LOG_DIR / "model_params.npy")
        self.models = {}
        self.feature_names = TRAINING_FEATURES
        self.load_weights()
    
    def load_weights(self):
        """
            Load the trained weights from file
        """
        try:
            all_params = np.load(self.weights_file, allow_pickle=True).item()
            
            for house, params_dict in all_params.items():
                self.models[house] = {
                    'W': np.array(params_dict['W']),
                    'b': params_dict['b']
                }
                
                if 'features' in params_dict:
                    self.feature_names = params_dict['features']
            
            print(f"Successfully loaded weights for {len(self.models)} houses.")
        except Exception as e:
            raise Exception(f"Error loading model weights: {e}")
    
    def sigmoid(self, z):
        """
            Sigmoid activation function
        """
        return 1 / (1 + np.exp(-z))
    
    def predict_probability(self, X):
        """
            Predict probabilities for each house
            
            :param X: Features DataFrame
            :return: dictionary of probabilities for each house
        """
        if params.standardize:
            X = standardize_with_saved_params(X, LOG_DIR / params.standardization_params)
    
        probabilities = {}
        for house, model in self.models.items():
            W = model['W']
            b = model['b']
            z = np.dot(X.values, W) + b
            probabilities[house] = self.sigmoid(z)
        return probabilities
    
    def predict(self, X):
        """
            Predict the house with highest probability for each student
            
            :param X: Features DataFrame
            :return: list of predicted houses
        """
        probabilities = self.predict_probability(X)
        predictions = [""] * len(X)
        
        for i in range(len(X)):
            best_house = None
            best_prob = -1
            
            for house, probs in probabilities.items():
                if probs[i] > best_prob:
                    best_prob = probs[i]
                    best_house = house
            predictions[i] = best_house
        return predictions

def standardize_with_saved_params(df: pd.DataFrame, stats_path: Path):
    """
        Using the standardization parameters of the training phase.
    """
    stats = pd.read_csv(stats_path, index_col=0)
    mean = stats['mean']
    std = stats['std']
    return (df - mean) / std

# ****************************************** MAIN ***************************************************


def main(parsed_args):
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading test data from \"{params.test_data_path}:\"")
    processor = DataPreProcessor(params.test_data_path)
    X_val = processor.pre_process()

    try:
        print("Making predictions...")
        predictor = LogisticRegressionPredictor()
        predictions = predictor.predict(X_val)
        save_predictions(predictions, LOG_DIR / "houses.csv")
    except Exception as e:
        raise Exception(f"Prediction logic error: {e}")
    
    if parsed_args.path_truth_csv:
        y_truth = processor.pre_process_truth(parsed_args.path_truth_csv)
        calculate_accuracy(predictions, y_truth)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_truth_csv',
                        type = str,
                        help = """Optional - Truth CSV file to read and calculate accuracy of the modele.""")
    parsed_args = parser.parse_args()
    main(parsed_args)