import pandas as pd
import numpy as np
from pathlib import Path
from utils.config import params
from utils.upload_csv import upload_csv
from utils.maths import MyMaths

LOG_DIR = params.LOG_DIR
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
    
    def standardize(self, df):
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
        """Predict probabilities for each house"""
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
        """Predict the house with highest probability for each student"""
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

def save_predictions(predictions, output_file):
    """Save predictions to a CSV file"""

    output = pd.DataFrame({'Hogwarts House': predictions})
    
    output.to_csv(output_file, index_label='Index')
    print(f"Predictions saved to {output_file}")

def main():
    
    LOG_DIR.mkdir(parents=True, exist_ok=True) 

    print(f"Loading test data from {params.test_data_path}...")
    data = upload_csv(params.test_data_path)
    
    try:        
        print("Preparing test data...")
        X = data[TRAINING_FEATURES].copy()
        X = X.dropna(subset=TRAINING_FEATURES)
        
        print(f"Loading model weights from {params.weights_file}...")
        predictor = LogisticRegressionPredictor()
        
        print("Making predictions...")
        predictions = predictor.predict(X)
        
        save_predictions(predictions, LOG_DIR / "predictions.csv")
        
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()