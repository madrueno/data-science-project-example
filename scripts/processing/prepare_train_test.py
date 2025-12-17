"""Script to prepare train and test sets from interim data."""

import pandas as pd
from sklearn.model_selection import train_test_split

from spam_ham_detector.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR


def load_and_select_columns() -> pd.DataFrame:
    """
    Load interim data and select most relevant columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with only CONTENT and CLASS columns
    """
    filepath = INTERIM_DATA_DIR / 'comments.csv'
    print(f"Loading {filepath}...")

    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Select only CONTENT and CLASS columns
    df_selected = df[['CONTENT', 'CLASS']].copy()
    print(f"\nSelected columns: {list(df_selected.columns)}")

    return df_selected


def split_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split data into train and test sets with stratification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with CONTENT and CLASS columns
    test_size : float
        Proportion of data to use for test set (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    tuple
        (train_df, test_df) DataFrames
    """
    print(f"\nSplitting data (test_size={test_size}, random_state={random_state})...")

    # Check class distribution
    class_counts = df['CLASS'].value_counts()
    print(f"Class distribution:\n{class_counts}")

    # Split with stratification to maintain class balance
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['CLASS']
    )

    print(f"\nTrain set: {len(train_df)} rows")
    print(f"Test set: {len(test_df)} rows")

    return train_df, test_df


def main() -> None:
    """
    Load comments.csv, select CONTENT and CLASS columns, and split into train/test.

    Reads from data/interim/comments.csv and saves:
    - data/processed/train.csv
    - data/processed/test.csv
    """
    # Load and select columns
    df = load_and_select_columns()

    # Split into train and test
    train_df, test_df = split_train_test(df)

    # Save train set
    train_path = PROCESSED_DATA_DIR / 'train.csv'
    print(f"\nSaving train set to {train_path}...")
    train_df.to_csv(train_path, index=False)

    # Save test set
    test_path = PROCESSED_DATA_DIR / 'test.csv'
    print(f"Saving test set to {test_path}...")
    test_df.to_csv(test_path, index=False)

    print("\nDone!")


if __name__ == '__main__':
    main()
