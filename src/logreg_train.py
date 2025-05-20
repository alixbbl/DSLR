import pandas as pd
import numpy as np
from utils.config import params
from sklearn.model_selection import train_test_split
import plotly.express as px
from utils.upload_csv import upload_csv
from utils.maths import MyMaths

LOG_DIR = params.LOG_DIR
HOGWART_HOUSES = params.hogwart_houses
TRAINING_FEATURES = params.training_features

class LogisticRegressionTrainer():
    
    def __init__(self, learning_rate = params.learning_rate, 
                 max_iterations = params.max_iterations,
                 X_train = None, 
                 y_train = None):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.optimization = params.optimization
        self.X = X_train
        self.y = y_train
        self.cost_history = []
        print(f"Optimization method: {self.optimization}")
    
    def initialize_parameter(self):
        """Initializes the parameters of the model."""
        self.W = np.zeros(self.X.shape[1])
        # print(self.W)
        self.b = 0.0
        # print(self.b)
        
    def sigmoid(self, z: np.ndarray):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
        
    def forward(self, X: np.ndarray):
        """Computes forward propagation for given input X."""
        Z = np.matmul(X, self.W) + self.b
        A = self.sigmoid(Z)
        return A
    
    def compute_cost(self, predictions: np.ndarray):
        """Compute binary cross-entropy loss"""
        m = self.X.shape[0]  
        cost = np.sum((-np.log(predictions + 1e-8) * self.y) + (-np.log(1 - predictions + 1e-8)) * (1 - self.y))
        cost = cost / m
        return cost
    
    def compute_gradient(self, predictions: np.ndarray):
        """Computes the gradients for the model using given predictions."""
        m = self.X.shape[0]
        # compute gradients
        self.dW = np.matmul(self.X.T, (predictions - self.y))
        self.db = np.sum(np.subtract(predictions, self.y))

        # scale gradients
        self.dW = self.dW * 1 / m
        self.db = self.db * 1 / m    

    def fit(self, X, y):
        """Mimics the fit function used as a builtin in ML libraries, this function will train the model using various 
            methods : gradient descent, stochastic gradient descent or mini-batch graident descent, as requested by user.
            :arguments : X => training features, y : target
            :returns : None
        """
        self.X = X
        self.y = y
        self.m = X.shape[0]

        self.initialize_parameter()
        for i in range(self.max_iterations):
            predictions = self.forward(self.X)

            cost = self.compute_cost(predictions)
            self.cost_history.append(cost)
            
            if self.optimization == "gradient_descent":
                self.compute_gradient(predictions)                
                cost = self.compute_cost(predictions)
                self.cost_history.append(cost)

            # BONUS : stochastic GD
            elif self.optimization == "stochastic_gradient_descent":
                random_index = np.random.randint(0, self.m)
                X_sample = self.X[random_index:random_index+1]
                y_sample = self.y[random_index:random_index+1]

                z_sample = np.dot(X_sample, self.W) + self.b
                prediction_sample = self.sigmoid(z_sample)

                self.dW = X_sample.T.dot(prediction_sample - y_sample)
                self.db = np.sum(prediction_sample - y_sample)

            # BONUS : mini-batch GD
            elif self.optimization == "mini_batch_gradient_descent":
                batch_size = params.batch_size
                num_batches = int(np.ceil(self.m / batch_size))
                
                if isinstance(self.X, pd.DataFrame):
                    shuffled_indices = np.random.permutation(self.m)
                    
                    for j in range(num_batches):
                        start_idx = j * batch_size
                        end_idx = min((j + 1) * batch_size, self.m)
                        
                        batch_indices = shuffled_indices[start_idx:end_idx]
                        
                        X_batch = self.X.iloc[batch_indices].values
                        y_batch = self.y.iloc[batch_indices].values
                        
                        X_orig, y_orig = self.X, self.y
                        
                        self.X, self.y = X_batch, y_batch
                        predictions_batch = self.forward(X_batch)
                        self.compute_gradient(predictions_batch)
                        
                        self.W = self.W - self.learning_rate * self.dW
                        self.b = self.b - self.learning_rate * self.db
                        
                        self.X, self.y = X_orig, y_orig
            else:
                raise ValueError("Invalid optimization method. Choose 'gradient_descent', 'stochastic_gradient_descent', or 'mini_batch_gradient_descent'.")
            
            self.W = self.W - self.learning_rate * self.dW
            self.b = self.b - self.learning_rate * self.db

            if i % 200 == 0:
                print("Cost after iteration {}: {}".format(i, cost))

# ************************************* DATE PREPROCESSING **************************************

def ft_standardize(df: pd.DataFrame) -> pd.DataFrame:
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
    return (df - mean) / std

def prepare_data(data: pd.DataFrame):
    """Clean and prepare data for training"""
    data = data.dropna(subset=TRAINING_FEATURES)
    X = data[TRAINING_FEATURES].copy()
    y = data['Hogwarts House'].copy()
    if params.standardize:
        X = ft_standardize(X)
    return (train_test_split(X, y, test_size=params.test_size, random_state=params.seed))
        
    
# ************************************* START / END  **************************************


def launch_trainer(X: pd.Series, y: pd.Series):
        """Train one-vs-all logistic regression models"""
        models = {}
        for house in HOGWART_HOUSES:
            print(f"\nTraining model for {house}...")
            y_binary = (y == house).astype(int) # va servir a encoder le 0 ou 1 de la classe
            model = LogisticRegressionTrainer(learning_rate=0.1, max_iterations=1000)
            model.fit(X, y_binary)
            models[house] = model 
        return models

def save_model_weights(models: dict, feature_names: list):
    print(feature_names)
    """Save model weights and parameters to files"""    
    all_params = {}
    
    for house, model in models.items():
        model_params = {
            'W': model.W.tolist(),
            'b': model.b,
            'learning_rate': model.learning_rate,
            'iterations': model.max_iterations,
            'features': feature_names
        }
        all_params[house] = model_params  
        with open(LOG_DIR / f"{house}_weights.txt", 'w') as f:
            f.write(f"Model weights for {house}:\n\n")
            f.write(f"Bias: {model.b}\n\n")
            f.write("Feature weights:\n")
            for i, feature in enumerate(feature_names):
                f.write(f"{feature}: {model.W[i]}\n")
            
            f.write(f"\nLearning rate: {model.learning_rate}\n")
            f.write(f"Iterations: {model.max_iterations}\n")
            f.write(f"Final cost: {model.cost_history[-1] if model.cost_history else 'N/A'}\n")
    np.save(LOG_DIR / "model_params.npy", all_params)
    print(f"Model weights saved to {LOG_DIR}")    


# *************************************** MAIN **************************************


def main():
        
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    data = upload_csv(params.training_data_path)
    
    try:
        print("Preparing data\n")
        X_train, X_test, y_train, y_test = prepare_data(data)
        print(y_test.head())

        print("Training models\n")
        models = launch_trainer(X_train, y_train)
        
        print("\nSaving model parameters")
        save_model_weights(models, TRAINING_FEATURES)
    
    except Exception as e:
        print(f'Something happened again: {e}')

if __name__ == "__main__":
    main()