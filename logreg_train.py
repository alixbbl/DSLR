import pandas as pd
import numpy as np
import argparse
from typing import List, Tuple
from utils.upload_csv import upload_csv
from utils.constants import EXPECTED_LABELS, MANDATORY_FEATURES_SET, TRAINING_FEATURES_LIST
from utils.maths import MyMaths
from utils.utils_logistic_regression import log_loss, write_output, plot_cost_report

class Trainer():
    
    def __init__(self, dataframe, max_it, learning_rate, relevant_feat):
        self.df = dataframe
        self.data_to_train = None # pd.DataFrame
        self.data_train_std = None # pd.Series
        self.data_train_mean = None  # pd.Series
        self.max_iterations = max_it
        self.learning_rate = learning_rate
        self.relevant_features = relevant_feat
        self.labels = set(self.df['Hogwarts House'].unique())
        if self.labels != EXPECTED_LABELS:
            raise ValueError('This is NOT an Hogwarts House !')

    def ft_is_valid_training_dataframe(self):
        """"
            This functions checks if the dataset is valid for training the model and contains
            the required training classes.
        """
        if ('Hogwarts House') not in self.df.columns:
            raise Exception('Sorry, this is not a proper training dataset')
        columns_list = list(self.df.columns)
        if not MANDATORY_FEATURES_SET <= columns_list:
            raise Exception('Magic hat needs more than that to perform its magic !')
        return True
    
    def ft_prepare_data(self) -> None:
        """"
            This function retuns a valid dataset for training by adding the default theta0 column, 
            required for scalar product and bias, and by deleting the null entries in the dataset (only 
            valid for a few null entries < 5%).
        """
        for feature in self.relevant_features:
            if self.df[feature].isnull().sum() > 0:
                print(f'For feature {feature} : {self.df[feature].isnull().sum()} NULL entries!')
            else:
                print(f'No null entry for {feature}')
        self.df.dropna(subset=self.relevant_features, inplace=True)
        
        self.data_to_train = self.df[self.relevant_features].copy() # on fait une copie plutot qu'une reference
        self.data_train_std, self.data_train_mean, self.data_to_train= self.ft_standardize_data(self.data_to_train)
        self.data_to_train.insert(0, "theta0", np.ones(self.data_to_train.shape[0])) # on ajoute une colonne de 1 pour le produit scalaire (a faire pour tous les cas de reg lin ou log)
        self.ft_encode_data() # on encode a la fin pour garder les binaires 0 et 1

    def ft_encode_data(self):
        """"
            This function transforms the categorical coloumn 'Hogwarts House' in a nunerical
            column to enhance the models training.
        """
        for feature in EXPECTED_LABELS: # on ajoute les 4 colonnes de 0 pour le one-hot-encoding
            self.data_to_train[feature] = [1 if ele == feature else 0 for ele in self.df['Hogwarts House']]
        # print(self.data_to_train)

    # Utilisation de apply() qui permet de calculer colonne a colonne sans boucle for
    def ft_standardize_data(self, mx: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """"
            This functions takes a matrix and returns a tuple of calculated elements : 
            the matrix std and mean, and the standardized matrix to operate. 
        """
        maths = MyMaths()
        mx_std = mx.apply(maths.my_std) # on applique la fonction a chaque colonne et retourne une Serie des std de chaque colonne
        mx_mean = mx.apply(maths.my_mean) # idem avec les moyennes de chaque col. apply() agir comme une boucle for
        mx_standardized = (mx - mx_mean) / mx_std
        return [mx_std, mx_mean, mx_standardized]
    
    # fonction utilitaire utilisee pour convertir un nombre en une probabilite de 0 a 1,
    # ici 0 est "faux" et 1 est "vrai" concernant une tache de classification.
    def ft_sigmoid(self, x):
        """Classic sigmoid function, converts any number in a 0 to 1 probability."""
        return 1 / (1 + np.exp(-x))
    
    # on remet la fonction issue de Linear Regression qui permet de calculer la descente de gradienbt et donc
    # de chercher le minimum de la fonction de cout (differente de celle de la RegLin)
    # Rappels : X est la matrice des features et y est la variable cible
    # m est le nombre d'entrees du dataset et n le nbre de features, cost_report stocke le loss a chaque iteration du training
    def ft_gradient_descent(self, X, y, max_iterations, learning_rate) -> Tuple[float, List[float]]:
    
        m, n = X.shape
        theta = np.zeros((n, 1))
        cost_report = []
        y = y.values.reshape(-1, 1)
    
        for i in range(max_iterations):
            y_predictions = self.ft_sigmoid(X.dot(theta)) # on multiplie la matrice des km normalises par le vecteur (0, 0), debut de la descente de gradient
            errors = y_predictions - y # on calcule l'erreur entre la matrice prediction et la matrice des prix reels du dataset
            gradient = (1 / m) * X.T.dot(errors) # formule du gradient
            theta -= learning_rate * gradient # formule d'actualisation de theta (theta0, theta1) l'intercept et la pente
            log_loss_value = log_loss(y, y_predictions) # injecter la fonction de log_loss qui est la fonction de cout en LogReg
            cost_report.append(log_loss_value)
        
            # if i % 100 == 0:
            #     print(f"Iteration {i}: Cost = {log_loss_value:.4f}, theta = {theta.T}")

        return theta, cost_report
        
    def ft_train(self) -> Tuple[float, List[float]]:
        """"This function performs a training for each of the houses."""
        # on realise le training pour chacune des maisons separement (cost loss function est binaire) -> a mettre dans ft_train()
        list_thetas = {}
        list_cost_reports = {}
        X = self.data_to_train[self.relevant_features]
        
        for house in EXPECTED_LABELS:
            y_house = self.data_to_train[house]
            list_thetas[house], list_cost_reports[house] = self.ft_gradient_descent(X, y_house, self.max_iterations, self.learning_rate)
        
        return list_thetas, list_cost_reports # list_thetas retourne les coeff par feature 

# *************************************** MAIN **************************************

def main(parsed_args):
    
    try:
        df = upload_csv(parsed_args.path_csv)
        if df is None: return
        try:
            trainer=Trainer(df, 
                            parsed_args.max_it, 
                            parsed_args.learning_rate,
                            TRAINING_FEATURES_LIST)      
            if trainer.ft_is_valid_training_dataframe():
                trainer.ft_prepare_data() # a ce stade on a un dataset d'entrainement fini (suppr. des nulls, encoding, standardization ...)
                list_thetas, list_cost_reports = trainer.ft_train()
                write_output(list_thetas)
                # Appelle la fonction de plot dans ton code où tu veux visualiser le rapport
                plot_cost_report(list_cost_reports['Gryffindor'])

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
    parser.add_argument('-i', '--max_it',
                        type=int,
                        help="""Max iterations to go through the regression.""")
    parser.add_argument('-l', '--learning_rate',
                        type=float,
                        help="""Learning rate of the model.""")    
    parsed_args=parser.parse_args()
    main(parsed_args)