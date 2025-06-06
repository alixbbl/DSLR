from pathlib import Path


class Config:

    BASE_DIR = Path(__file__).parent.absolute()
    # paths
    LOG_DIR = BASE_DIR / "output" / "log"
    DATA_DIR = BASE_DIR.parent / "data"

    training_data_path = DATA_DIR / "dataset_train.csv"
    test_data_path = DATA_DIR / "dataset_test.csv"
    validation_data_path_features = DATA_DIR / "x_dataset_validation.csv"
    validation_data_path_target = DATA_DIR / "y_dataset_validation.csv"

    weights_file = LOG_DIR / "model_params.npy"
    standardization_params_path = LOG_DIR / "standardization_params.csv"

    # data
    hogwart_houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    training_features = [
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Ancient Runes",
        "Charms",
        "Transfiguration",
    ]

    standardize = True

    # hyperparameters
    learning_rate = 0.1
    epochs = 1000
    batch_size = 32
    seed = 42
    test_size = 0.2

    # model
    # optimization = "gradient_descent"
    optimization = "stochastic_gradient_descent"
    # optimization = "mini_batch_gradient_descent"


params = Config()
