from pathlib import Path


class Config():

    # paths
    LOG_DIR = Path("output/logreg")
    DATA_DIR = Path("data")
    training_data_path = "data/dataset_train.csv"
    validation_data_path = "my_validation_dataset.csv"
    test_data_path = "data/dataset_test.csv"
    weights_file = LOG_DIR / "model_params.npy"

    # data
    hogwart_houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    training_features = ["Defense Against the Dark Arts", "Herbology", "Ancient Runes", "Astronomy", "Transfiguration"]
    # training_features = [
    #                         'Astronomy', 
    #                         'Herbology', 
    #                         'Divination', 
    #                         'Muggle Studies', 
    #                         'Ancient Runes', 
    #                         'History of Magic',
    #                         'Charms', 
    #                         ]
    standardize = True

    #hyperparameters
    learning_rate = 0.1
    max_iterations = 1000
    batch_size = 32
    seed = 42
    test_size = 0.2

    # model
    optimization = "gradient_descent" # "stochastic_gradient_descent", "mini_batch_gradient_descent" or "gradient_descent"


params = Config()