"""Dataset loader for YouTube comment spam classification."""

import os
import pandas as pd


class CommentsDataset:
    """
    Loader for YouTube comment spam classification datasets.

    Loads train and test CSV files from the processed data directory
    and splits them into features (X) and labels (y).

    Attributes
    ----------
    X_train : pd.Series
        Training features (comment text)
    y_train : pd.Series
        Training labels (0=ham, 1=spam)
    X_test : pd.Series
        Test features (comment text)
    y_test : pd.Series
        Test labels (0=ham, 1=spam)
    """

    def __init__(self):
        """
        Load datasets and split into features and labels.

        Parameters
        ----------
        data_dir : str, optional
            Path to the processed data directory (default: 'data/processed')
        """
        # Load datasets
        train_path = os.path.join('data', 'processed', 'train.csv')
        test_path = os.path.join('data', 'processed', 'test.csv')

        train_df = pd.read_csv(train_path, encoding='utf-8')
        test_df = pd.read_csv(test_path, encoding='utf-8')

        # Split into features and labels
        self.X_train = train_df['CONTENT']
        self.y_train = train_df['CLASS']
        self.X_test = test_df['CONTENT']
        self.y_test = test_df['CLASS']


def main() -> None:
    """Load datasets and display class distributions."""
    dataset = CommentsDataset()

    print(f"Train set: {len(dataset.X_train)} rows")
    print(f"Test set: {len(dataset.X_test)} rows")

    print("\nTrain class distribution:")
    print(dataset.y_train.value_counts())

    print("\nTest class distribution:")
    print(dataset.y_test.value_counts())


if __name__ == '__main__':
    main()
