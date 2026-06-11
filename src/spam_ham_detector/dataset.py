"""Dataset loader for YouTube comment spam classification."""

import pandas as pd

from spam_ham_detector.config import PROCESSED_DATA_DIR


class CommentsDataset:
    """
    Loader for YouTube comment spam classification datasets.

    Attributes
    ----------
    X_train : list[str]
        Training features (comment text)
    y_train : list[int]
        Training labels (0=ham, 1=spam)
    X_dev : list[str]
        Dev/validation features
    y_dev : list[int]
        Dev/validation labels
    X_test : list[str]
        Test features (comment text)
    y_test : list[int]
        Test labels (0=ham, 1=spam)
    """

    def __init__(self):
        """Load train/dev/test splits from precomputed files."""
        train_df = pd.read_csv(PROCESSED_DATA_DIR / 'train.csv')
        dev_df = pd.read_csv(PROCESSED_DATA_DIR / 'dev.csv')
        test_df = pd.read_csv(PROCESSED_DATA_DIR / 'test.csv')

        self.X_train = train_df['CONTENT'].to_list()
        self.y_train = train_df['CLASS'].to_list()

        self.X_dev = dev_df['CONTENT'].to_list()
        self.y_dev = dev_df['CLASS'].to_list()

        self.X_test = test_df['CONTENT'].to_list()
        self.y_test = test_df['CLASS'].to_list()
