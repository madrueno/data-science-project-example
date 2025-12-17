"""Dataset loader for YouTube comment spam classification."""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


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
    X_dev : pd.Series or None
        Dev/validation features (created by get_dev_split())
    y_dev : pd.Series or None
        Dev/validation labels (created by get_dev_split())
    X_test : pd.Series
        Test features (comment text)
    y_test : pd.Series
        Test labels (0=ham, 1=spam)
    """

    def __init__(self):
        """Load train/test/dev splits automatically."""
        # Load datasets
        train_path = os.path.join('data', 'processed', 'train.csv')
        test_path = os.path.join('data', 'processed', 'test.csv')

        train_df = pd.read_csv(train_path, encoding='utf-8')
        test_df = pd.read_csv(test_path, encoding='utf-8')

        # Test set
        self.X_test = test_df['CONTENT']
        self.y_test = test_df['CLASS']

        # Split train into train/dev (dev same size as test)
        X_train_full = train_df['CONTENT']
        y_train_full = train_df['CLASS']

        dev_size = len(self.X_test) / len(X_train_full)

        self.X_train, self.X_dev, self.y_train, self.y_dev = train_test_split(
            X_train_full,
            y_train_full,
            test_size=dev_size,
            random_state=42,
            stratify=y_train_full
        )
