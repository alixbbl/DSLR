def calculate_accuracy(y_pred, y_true) -> float:
    """
    Calculate the accuracy of the model on test data

    :param y_pred: Calculated predictions
    :param y_true: True labels (Hogwarts Houses)
    :return: accuracy score (0-1)
    """
    correct = sum(1 for pred, true in zip(y_pred, y_true) if pred == true)
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy