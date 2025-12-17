"""Evaluation metrics for classification models."""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def evaluate_model(model, X, y_true, split_name="test"):
    """
    Evaluate classification model and print metrics.

    Parameters
    ----------
    model : sklearn classifier
        Trained classifier with predict() method
    X : array-like or sparse matrix
        Feature matrix
    y_true : array-like
        True labels
    split_name : str
        Name of the split (e.g., 'train', 'dev', 'test')

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, and f1 scores
    """
    y_pred = model.predict(X)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Print results
    print(f"\n=== {split_name.upper()} SET RESULTS ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['ham', 'spam']))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
