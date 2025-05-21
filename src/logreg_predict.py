import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from utils.config import params
from utils.upload_csv import upload_csv
from utils.metrics import calculate_accuracy
from utils.maths import MyMaths
from logreg_validation import calculate_accuracy

LOG_DIR = params.LOG_DIR
DATA_DIR = params.DATA_DIR
HOGWART_HOUSES = params.hogwart_houses
TRAINING_FEATURES = params.training_features

class LogisticRegressionPredictor:
    def __init__(self, weights_file=None):
        self.weights_file = params.weights_file if weights_file else str(LOG_DIR / "model_params.npy")
        self.models = {}
        self.feature_names = TRAINING_FEATURES
        self.load_weights()
    
    def load_weights(self):
        """Load the trained weights from file"""
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
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def standardize(self, df: pd.DataFrame):
        """
        Standardize features to mean=0 and std=1.
        
        :param df: DataFrame
        :return: standardized DataFrame
        """
        maths = MyMaths()
        std = df.apply(maths.my_std) 
        mean = df.apply(maths.my_mean) 
        return (df - mean) / std
    
    def predict_proba(self, X):
        """
        Predict probabilities for each house
        
        :param X: Features DataFrame
        :return: dictionary of probabilities for each house
        """
        if params.standardize:
            X = self.standardize(X)
    
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
        probabilities = self.predict_proba(X)
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

def ft_imputation_by_mean(df: pd.DataFrame) -> pd.DataFrame:

    prediction_dataset = pd.DataFrame()
    for feature in df.columns:
        prediction_dataset[feature] = df[feature]
    prediction_dataset.fillna(prediction_dataset.mean(), inplace=True) # imputation par la moyenne
    return prediction_dataset

def standardize_with_saved_params(df: pd.DataFrame, stats_path: Path):
    stats = pd.read_csv(stats_path, index_col=0)
    mean = stats['mean']
    std = stats['std']
    return (df - mean) / std

def save_predictions(predictions, output_file):
    """
    Save predictions to a CSV file
    
    :param predictions: list of predicted houses
    :param output_file: path to output CSV file
    """
    output = pd.DataFrame({'Hogwarts House': predictions})
    output.to_csv(output_file, index_label='Index')
    print(f"Predictions saved to {output_file}")


# ****************************************** MAIN ***************************************************


def main(parsed_args):
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading test data from {params.test_data_path}...")
    data_test = upload_csv(params.test_data_path)
    
    try:        
        print("Preparing test data...")
        X = data_test[TRAINING_FEATURES].copy()
        try:
            X = ft_imputation_by_mean(X)
        except Exception as e:
            raise Exception(f"Preparing data error: {e}")

        if params.standardize:
            X = standardize_with_saved_params(X, LOG_DIR / "standardization_params.csv")
        print(f"Loading model weights from {params.weights_file}...")
        
        try:
            predictor = LogisticRegressionPredictor()
            print("Making predictions...")
            predictions = predictor.predict(X)
            save_predictions(predictions, LOG_DIR / "houses.csv")
        except Exception as e:
            raise Exception(f"Prediction logic error: {e}")
        
        # calcul de perfomance si le fichier de verites terrain est fourni : 
        if parsed_args.path_truth_csv:
            data_truth = upload_csv(parsed_args.path_truth_csv)
            if 'Index' in data_truth.columns:
                data_truth.drop(columns='Index', inplace=True)
            y_truth = data_truth['Hogwarts House']
            y_pred = predictions
            try :
                accuracy = calculate_accuracy(y_pred, y_truth)
                print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            except:
                raise Exception(f'Calculating accuracy failure : {e}')
    
    except Exception as e:
        print(f"Error during prediction: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_truth_csv',
                        type = str,
                        default = None,
                        help = """Optionnal - Truth CSV file to read and calculate accuracy of the modele.""")
    parsed_args = parser.parse_args()
    main(parsed_args)
