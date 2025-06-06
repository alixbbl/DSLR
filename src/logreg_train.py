import pandas as pd
import numpy as np
from utils.config import params
from sklearn.model_selection import train_test_split
# import plotly.express as px
import tensorflow as tf
from utils.upload_csv import upload_csv
from utils.maths import MyMaths
from utils.store import store_df_to_csv
from typing import Tuple
import matplotlib.pyplot as plt
from utils.tensorboard import TensorBoardCallback

LOG_DIR = params.LOG_DIR
DATA_DIR = params.DATA_DIR

class LogisticRegressionTrainer():
    
    def __init__(self, learning_rate: float = 0.1, 
                 epochs: int = 100,
                 optimization:str = "gradient_descent"):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimization = optimization
        self.cost_history = []
        print(f"Optimization method: {self.optimization}")

    ## ------ LOG and INIT -----

    def log_progress(self, epoch: int, cost: float, callback: TensorBoardCallback, house_name: str):
        if callback:
            callback.log_iteration(epoch, cost, house_name)
        if epoch % 200 == 0 or epoch == self.epochs - 1:
            print(f"Cost after iteration {epoch}: {cost}")

    def initialize_parameter(self):
        """
            Initializes the parameters of the model.
        """
        self.W = np.zeros(self.X.shape[1])
        self.b = 0.0

    ## ------ OPTIMIZERS -----

    def perform_optimization_step(self):
        """Perform one optimization step based on the selected method."""
        if self.optimization == "gradient_descent":
            return self.gradient_descent_step()
        elif self.optimization == "stochastic_gradient_descent":
            return self.stochastic_gradient_descent_step()
        elif self.optimization == "mini_batch_gradient_descent":
            return self.mini_batch_gradient_descent_step()
        else:
            raise ValueError(
                "Invalid optimization method. Choose 'gradient_descent', "
                "'stochastic_gradient_descent', or 'mini_batch_gradient_descent'."
            )

    def gradient_descent_step(self):
        """Perform one gradient descent step using the full dataset."""
        predictions = self.forward(self.X)
        dW, db = self.compute_gradient(self.X, predictions, self.y)
        self.update_parameters(dW, db)
        return predictions
    
    def stochastic_gradient_descent_step(self):
        """Perform one SGD step using a single random sample."""
        random_index = np.random.randint(0, self.m)
        X_sample = self.X[random_index:random_index+1]
        y_sample = self.y[random_index:random_index+1]
        
        predictions = self.forward(X_sample)
        dW, db = self.compute_gradient(X_sample, predictions, y_sample)
        self.update_parameters(dW, db)
        
        return self.forward(self.X)
    
    def mini_batch_gradient_descent_step(self):
        """Perform one mini-batch gradient descent step."""
        batch_size = params.batch_size
        
        for batch_indices in self.get_batch_indices(batch_size):
            if isinstance(self.X, pd.DataFrame):
                X_batch = self.X.iloc[batch_indices].values
                y_batch = self.y.iloc[batch_indices].values
            else:
                X_batch = self.X[batch_indices]
                y_batch = self.y[batch_indices]
            
            predictions_batch = self.forward(X_batch)
            dW, db = self.compute_gradient(X_batch, predictions_batch, y_batch)
            self.update_parameters(dW, db)
        
        # Return full dataset predictions for cost calculation
        return self.forward(self.X)
    
    def get_batch_indices(self, batch_size: int):
        """Generate batch indices for mini-batch processing."""
        shuffled_indices = np.random.permutation(self.m)
        
        for start_idx in range(0, self.m, batch_size):
            end_idx = min(start_idx + batch_size, self.m)
            yield shuffled_indices[start_idx:end_idx]

    ## ------ COMPUTES -----

    def sigmoid(self, z: np.ndarray):
        """
            Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-z))
        
    def forward(self, X: np.ndarray):
        """
            Computes forward propagation for given input X.
        """
        Z = np.matmul(X, self.W) + self.b
        A = self.sigmoid(Z)
        return A
    
    def compute_cost(self, predictions: np.ndarray, y: np.ndarray):
        """Compute binary cross-entropy loss."""
        m = len(y)
        cost = np.sum((-np.log(predictions + 1e-8) * y) + 
                     (-np.log(1 - predictions + 1e-8)) * (1 - y))
        cost = cost / m
        return cost
    
    def compute_gradient(self, X: np.ndarray, predictions: np.ndarray, y: np.ndarray):
        """Compute gradients for the model using given predictions."""
        m = len(y)
        dW = np.matmul(X.T, (predictions - y)) / m
        db = np.sum(predictions - y) / m
        return dW, db
    
    def update_parameters(self, dW: np.ndarray, db: float):
        """Update model parameters using gradients."""
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    ## ------ FIT ----- 

    def fit(self, X, y, callback, house_name):
        """
            Train the model using the selected optimization method.
        """
        self.X = X
        self.y = y
        self.m = X.shape[0]
        self.initialize_parameter()

        for epoch in range(self.epochs):
            all_predictions = self.perform_optimization_step()
            cost = self.compute_cost(all_predictions, self.y)
            self.cost_history.append(cost)
            self.log_progress(epoch, cost, callback, house_name)


# ************************************* DATA PREPROCESSING **************************************

def ft_standardize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
        Standardize features to mean=0 and std=1.
        :param df: DataFrame
        :return: standardized DataFrame
    """
    maths = MyMaths()
    std = df.apply(maths.my_std) 
    mean = df.apply(maths.my_mean)
    stats = pd.DataFrame({'mean': mean, 'std': std})
    stats.to_csv(LOG_DIR / "standardization_params.csv")
    return (df - mean) / std, mean, std

def apply_standardization(df: pd.DataFrame, mean: pd.Series, std: pd.Series)-> pd.DataFrame:
    """
        Takes mean and std of the X_train dataset and applies to another dataset(X_val, ...).
    """
    return (df - mean) / std

def prepare_data(data: pd.DataFrame):
    """
        Clean and prepare data for training, splitting must be applied before standardization to avoid any data leakage.
        arguments: Dataset.
        returns: 2 datasets for training and 2 for model validation.
    """
    data = data.dropna(subset=params.training_features)
    X = data[params.training_features].copy()
    y = data['Hogwarts House'].copy()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=params.test_size, random_state=params.seed)
    
    if params.standardize:
        X_train, mean, std = ft_standardize(X_train)
        X_val = apply_standardization(X_val, mean, std)

    return (X_train, X_val, y_train, y_val)
        
    
# ************************************* START / END  **************************************


def save_model_weights(models: dict, feature_names: list):
    """
        Save model weights and parameters to files
    """    
    print(feature_names)
    all_params = {}
    
    for house, model in models.items():
        W = model.W.values if isinstance(model.W, pd.Series) else model.W
        model_params = {
            'W': W.tolist(),
            'b': model.b,
            'learning_rate': model.learning_rate,
            'epochs': model.epochs,
            'features': feature_names
        }
        all_params[house] = model_params  
        
        with open(LOG_DIR / f"{house}_weights.txt", 'w') as f:

            f.write(f"Model weights for {house}:\n\n")
            f.write(f"Bias: {model.b}\n\n")
            f.write("Feature weights:\n")
            for i, feature in enumerate(feature_names):
                f.write(f"{feature}: {W[i]}\n")
            f.write(f"\nLearning rate: {model.learning_rate}\n")
            f.write(f"Epochs: {model.epochs}\n")
            f.write(f"Final cost: {model.cost_history[-1] if model.cost_history else 'N/A'}\n")
    
    np.save(LOG_DIR / "model_params.npy", all_params)
    print(f"Model weights saved to {LOG_DIR}")
  
# ******************************* DATA VISUALIZATION ********************************

def plot_costs(models, LOG_DIR):

    for house, model in models.items():
        plt.plot(model.cost_history, label=house)

    plt.xlabel('Epochs')
    plt.ylabel('Cost (Loss)')
    plt.title(f'Cost function over epochs ({params.optimization})')
    plt.legend()
    plt.grid(True)
    filepath = LOG_DIR / f'Model_cost_figures({params.optimization}).png'
    plt.savefig(filepath)
    plt.show()


# *************************************** MAIN **************************************


def main():
        
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    callback = TensorBoardCallback(LOG_DIR)
    data = upload_csv(params.training_data_path_features)
    
    try:
        print("Preparing data for training...\n")
        X_train, X_val, y_train, y_val = prepare_data(data)

        # save validation sets
        store_df_to_csv(X_val, "my_validation_dataset_features", DATA_DIR)
        store_df_to_csv(y_val, "my_validation_dataset_target", DATA_DIR)

        print("Training models...\n")
        models = {}
        for house in params.hogwart_houses:
            print(f"\nTraining model for {house}...")
            y_binary = (y_train == house).astype(int) 
            model = LogisticRegressionTrainer(params.learning_rate, params.epochs, params.optimization)
            model.fit(X_train, y_binary, callback, house)
            models[house] = model 

        print("\nSaving model parameters...")
        save_model_weights(models, params.training_features)
        plot_costs(models, LOG_DIR)
    
    except Exception as e:
        print(f'Something happened: {e}')

if __name__ == "__main__":
    main()