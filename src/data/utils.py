from typing import List, Tuple

from pandas import DataFrame

def train_validation_test_split(
        assets:List[DataFrame],
        ratio:Tuple[float]=(0.7, 0.15, 0.15)
) -> (List[DataFrame], List[DataFrame], List[DataFrame]):
    """ Splits timeseries of assets into train, validation and test sets.

    Args:
        assets: List of timeseries DataFrames of asset features
        ratio: Split ratio (train, validation, test)

    Returns:
        3-Tuple with Lists of timeseries DataFrames: (list of training
        timeframes of each asset, list of validation timeframes, list of testing
        timeframes)
    """
    pass