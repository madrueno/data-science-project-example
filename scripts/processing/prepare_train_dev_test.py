"""Script to clean and split interim data into train, dev, and test sets."""

import pandas as pd
from sklearn.model_selection import train_test_split

from spam_ham_detector.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR


def load_and_clean() -> pd.DataFrame:
    """
    Load interim data and apply cleaning steps.

    All columns are kept so the splits saved to data/processed/ retain
    metadata (AUTHOR, DATE) for exploratory analysis in eda.ipynb.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with all columns.
    """
    filepath = INTERIM_DATA_DIR / 'comments.csv'
    print(f'Loading {filepath}...')

    df = pd.read_csv(filepath, encoding='utf-8')
    print(f'Total rows: {len(df)}')
    print(f'Columns: {list(df.columns)}')
    print('-' * 40)

    # Remove malformed rows (containing newlines - indicates CSV parsing errors)
    initial_rows = len(df)
    df = df[~df['CONTENT'].str.contains('\n', na=False)]
    print(f'Removed {initial_rows - len(df)} malformed rows')

    # Remove duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['CONTENT'], keep='first')
    print(f'Removed {initial_rows - len(df)} duplicate rows')
    print(f'Clean dataset: {len(df)} rows')
    print('-' * 40)

    return df


def split_train_dev_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split data into train, dev, and test sets with stratification.

    Dev is carved from the train set and kept the same size as test.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with all columns (COMMENT_ID, AUTHOR, DATE, CONTENT, CLASS)
    test_size : float
        Proportion of data to use for test set (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    tuple
        (train_df, dev_df, test_df) DataFrames
    """
    print(f'Splitting data (test_size={test_size}, random_state={random_state})...')
    print(df['CLASS'].value_counts().to_string())

    train_full, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['CLASS'])

    train_df, dev_df = train_test_split(
        train_full, test_size=len(test_df), random_state=random_state, stratify=train_full['CLASS']
    )

    print(f'Train set: {len(train_df)} rows')
    print(f'Dev set: {len(dev_df)} rows')
    print(f'Test set: {len(test_df)} rows')
    print('-' * 40)

    return train_df, dev_df, test_df


def main() -> None:
    """
    Load and clean interim/comments.csv, then split into train/dev/test.

    Each split is saved to data/processed/{split}.csv with all columns
    (COMMENT_ID, AUTHOR, DATE, CONTENT, CLASS). These are the canonical
    datasets for modeling and downstream analysis.
    """
    df = load_and_clean()

    # Split into train/dev/test
    train_df, dev_df, test_df = split_train_dev_test(df)

    # Save train set
    train_path = PROCESSED_DATA_DIR / 'train.csv'
    print(f'Saving train set to {train_path}...')
    train_df.to_csv(train_path, index=False)

    # Save dev set
    dev_path = PROCESSED_DATA_DIR / 'dev.csv'
    print(f'Saving dev set to {dev_path}...')
    dev_df.to_csv(dev_path, index=False)

    # Save test set
    test_path = PROCESSED_DATA_DIR / 'test.csv'
    print(f'Saving test set to {test_path}...')
    test_df.to_csv(test_path, index=False)

    print('-' * 40)
    print('Done!')


if __name__ == '__main__':
    main()
