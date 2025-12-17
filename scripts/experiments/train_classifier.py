"""Training pipeline for spam classification."""

from spam_ham_detector.dataset import CommentsDataset
from spam_ham_detector.evaluation import evaluate_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def create_pipeline():
    """
    Create text vectorizer and classifier.

    Returns
    -------
    tuple
        (vectorizer, classifier) ready for training
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    return vectorizer, classifier


def train_model(dataset):
    """
    Train spam classifier on dataset.

    Parameters
    ----------
    dataset : CommentsDataset
        Dataset with train/dev/test splits

    Returns
    -------
    tuple
        (vectorizer, classifier) trained on the dataset
    """
    print("Creating pipeline...")
    vectorizer, classifier = create_pipeline()

    print("Vectorizing text...")
    X_train_vec = vectorizer.fit_transform(dataset.X_train)
    X_dev_vec = vectorizer.transform(dataset.X_dev)
    X_test_vec = vectorizer.transform(dataset.X_test)

    print("Training classifier...")
    classifier.fit(X_train_vec, dataset.y_train)

    print("\nEvaluating model...")
    evaluate_model(classifier, X_train_vec, dataset.y_train, split_name="train")
    evaluate_model(classifier, X_dev_vec, dataset.y_dev, split_name="dev")
    evaluate_model(classifier, X_test_vec, dataset.y_test, split_name="test")

    return vectorizer, classifier


def main():
    """Load dataset, train classifier, and evaluate."""
    print("Loading dataset...")
    dataset = CommentsDataset()

    print(f"Train set: {len(dataset.X_train)} rows")
    print(f"Dev set: {len(dataset.X_dev)} rows")
    print(f"Test set: {len(dataset.X_test)} rows")

    vectorizer, classifier = train_model(dataset)

    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == '__main__':
    main()
