import pandas as pd
import numpy as np
import argparse
from typing import List, Tuple
from utils.upload_csv import upload_csv
from utils.constants import TRAINING_FEATURES_LIST, USELESS_COLUMNS_PREDICTING_PHASE
from utils.utils_logistic_regression import write_output, plot_cost_report

class Tester():

    def __init__(self, dataframe, thetas):
        self.df = dataframe
        self.prediction_dataset = pd.DataFrame()
        self.thetas = thetas
    
    def ft_is_valid_testing_dataframe(self):
        """"
            This functions checks if the dataset is valid for testing the model and contains
            the required testing classes.
        """
        columns_list = list(self.df.columns)
        if not TRAINING_FEATURES_LIST <= columns_list:
            raise Exception('Magic hat needs more than that to perform its magic ! Features must be similar !')
        restrained_dataset = self.df[TRAINING_FEATURES_LIST]
        for feature, expected_feature_type in zip(TRAINING_FEATURES_LIST, restrained_dataset.dtypes):
            if restrained_dataset[feature].dtype != expected_feature_type:
                raise Exception(f'Feature {feature} type in testing must be the same as in training !')    
        return True
        
    def ft_prepare_prediction_data(self):
        self.df = self.df.drop(columns=USELESS_COLUMNS_PREDICTING_PHASE)
        for feature in TRAINING_FEATURES_LIST:
            self.prediction_dataset[feature] = self.df[feature]
        # Ici on ne va pas pouvoir supprimer les donnees liees a des null entries => imputation
        # for feature in TRAINING_FEATURES_LIST:
        #     if self.prediction_dataset[feature].isnull().sum() > 0:
        #         print(f'For feature {feature} : {self.df[feature].isnull().sum()} NULL entries!')
        #     else:
        #         print(f'No null entry for {feature}')
        
    
# *************************************** MAIN **************************************

def main(parsed_args):
    
    try:
        df = upload_csv(parsed_args.path_csv)
        thetas = upload_csv(parsed_args.thetas_file)
        if df is None or thetas is None: return
        try:
            tester=Tester(df, thetas)      
            if tester.ft_is_valid_testing_dataframe():
                tester.ft_prepare_prediction_data() # a ce stade on a un dataset d'entrainement fini (suppr. des nulls, encoding, standardization ...)
                # list_thetas, list_cost_reports = tester.ft_predict()
                # output => efficacite du modele
                # visualisation ?

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
    parser.add_argument('-t', '--thetas_file',
                        nargs='?',
                        type=str,
                        help="""Path of CSV file containing the thetas for predictions.""")
    parsed_args=parser.parse_args()
    main(parsed_args)