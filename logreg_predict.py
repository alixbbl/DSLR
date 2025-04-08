import pandas as pd
import numpy as np
import argparse
from typing import List, Tuple
from utils.upload_csv import upload_csv
from utils.constants import TRAINING_FEATURES_LIST, USELESS_COLUMNS_PREDICTING_PHASE
from utils.utils_logistic_regression import write_output_predictions

class Tester():

    def __init__(self, dataframe, thetas, standard_constants):
        self.df = dataframe
        self.prediction_dataset = pd.DataFrame()
        self.thetas = thetas
        self.constants_stand = standard_constants
    
    def ft_is_valid_testing_dataframe(self):
        """"
            This functions checks if the dataset is valid for testing the model and contains
            the required testing classes.
        """
        columns_set = set(self.df.columns)
        if not set(TRAINING_FEATURES_LIST) <= columns_set:
            raise Exception('Magic hat needs more than that to perform its magic ! Features must be similar !')
        restrained_dataset = self.df[TRAINING_FEATURES_LIST]
        for feature, expected_feature_type in zip(TRAINING_FEATURES_LIST, restrained_dataset.dtypes):
            if restrained_dataset[feature].dtype != expected_feature_type:
                raise Exception(f'Feature {feature} type in testing must be the same as in training !')    
        return True
    
    def ft_standardize_data(self) -> None:
        """
        Standardize the prediction dataset using the constants in self.constants_stand.
        """
        constants = self.constants_stand.set_index("Feature")
        mx_mean = constants["Mean"]
        mx_std = constants["Std"]
        features_to_standardize = [f for f in self.prediction_dataset.columns if f in mx_mean.index]
        for feature in features_to_standardize:
            self.prediction_dataset[feature] = (
                self.prediction_dataset[feature] - mx_mean[feature]
            ) / mx_std[feature]

    def ft_prepare_prediction_dataset(self):
        self.df = self.df.drop(columns=USELESS_COLUMNS_PREDICTING_PHASE)
        for feature in TRAINING_FEATURES_LIST:
            self.prediction_dataset[feature] = self.df[feature]
        self.prediction_dataset.fillna(self.prediction_dataset.mean(), inplace=True) # imputation par la moyenne
        self.ft_standardize_data()

    def ft_stable_sigmoid(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def ft_predict(self):
        X = self.prediction_dataset.copy()
        X.insert(0, 'Bias', 1)
        X = X.to_numpy(dtype=float)

        thetas_df = self.thetas.copy()
        thetas_df = thetas_df.set_index("Hogwarts House")
        thetas_df.insert(0, "Bias", 0.0)

        features = list(thetas_df.columns)
        thetas = thetas_df[features].to_numpy(dtype=float)

        Z = np.dot(X, thetas.T)
        probas = self.ft_stable_sigmoid(Z)
        hogwarts_houses = list(thetas_df.index)
        predicted_classes_indices = np.argmax(probas, axis=1)
        predictions = [hogwarts_houses[i] for i in predicted_classes_indices]

        return predictions


# *************************************** MAIN **************************************

def main(parsed_args):
    try:
        df = upload_csv(parsed_args.path_csv)
        thetas = upload_csv(parsed_args.thetas_file)
        standards = upload_csv(parsed_args.standard_file)
        if df is None or thetas is None: return
        try:
            tester=Tester(df, thetas, standards)
            if tester.ft_is_valid_testing_dataframe():
                tester.ft_prepare_prediction_dataset()
                predictions = tester.ft_predict()
            write_output_predictions(predictions)

        except Exception as e:
            print(f'Something happened : {e}')
    except Exception as e:
        print(f'Something happened again: {e}')

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-p', '--path_csv',
                        nargs='?',
                        type=str,
                        help="""Path of CSV file to read""")
    parser.add_argument('-s', '--standard_file',
                        nargs='?',
                        type=str,
                        help="""Path of CSV file containing the constants to standardize.""")
    parser.add_argument('-t', '--thetas_file',
                        nargs='?',
                        type=str,
                        help="""Path of CSV file containing the thetas for predictions.""")
    parsed_args=parser.parse_args()
    main(parsed_args)