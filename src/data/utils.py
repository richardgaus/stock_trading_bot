from typing import List, Tuple
from copy import deepcopy

import warnings
import numpy as np
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
        Tuple of three lists: Lists of training, validation, and test set
        timeseries DataFrames
    """
    # Distribute test set ratio unevenly among assets so that test and
    # validation sets do not line up chronologically
    n = len(assets)
    distribute = []
    whole = 1.0
    for i in range(n):
        if i == n - 1:
            distribute.append(whole)
        else:
            std = 1 / (3 * (n - i))
            loc = 1 / (n - i)
            share = whole * np.clip(
                np.random.normal(loc, std), 0.5 * loc, 1.5 * loc
            )
            distribute.append(share)
            whole -= share

    # Split each asset dataframe
    train_list, valid_list, test_list = [], [], []
    for i, asset in enumerate(assets):
        test_beginning = 1.0 - distribute[i] * ratio[2] * n
        valid_beginning = test_beginning - ratio[1]
        test_beginning = int(test_beginning * len(asset))
        valid_beginning = int(valid_beginning * len(asset))
        train_list.append(asset[:valid_beginning])
        valid_list.append(asset[valid_beginning:test_beginning])
        test_list.append(asset[test_beginning:])

    return (train_list, valid_list, test_list)


def make_Xy(timeseries:DataFrame,
            features:List[str],
            outputs:List[str],
            num_timesteps:int=365,
            future_interval:Tuple[int]=(1, 3, 7),
            standardize:bool=False
            ) -> (np.array, np.array):
    """ Create X and y dataframes from raw timeseries data. Most recent
    timesteps must be most upward in dataframe.

    Args:
        timeseries: Timeseries dataframe
        features: Features for prediction
        outputs:
        num_timesteps:
        future_interval:

    Returns:
        X, y for training of neural networks.
    """
    if standardize:
        if not set(outputs).issubset(features):
            raise ValueError('When standardizing, all outputs must be in the features as well, '
                             'so the outputs can be standardized using parameters acquired from'
                             ' inputs.')
        outputs_idx = np.array([
            features.index(out) for out in outputs
        ])
    ts_reverse = timeseries.iloc[::-1]
    X_list = []
    y_list = []
    t = num_timesteps - 1
    while t < len(ts_reverse) - max(future_interval):
        X_list.append(
            ts_reverse.iloc[t - num_timesteps + 1:t + 1][features].transpose().to_numpy()
        )
        y_list.append(
            ts_reverse.iloc[[t + f for f in future_interval]][outputs].transpose().to_numpy()
        )
        t += 1

    if len(X_list) == 0:
        warnings.warn('X, y are empty. Use lower num_timesteps or '
                      'future_interval or supply longer timeseries.')
        X = np.array(None)
        y = np.array(None)
    else:
        X = np.stack(X_list)
        y = np.stack(y_list)
        if standardize and len(X_list) > 0:
            # Standardize X and y with parameters acquired on X
            means = np.expand_dims(np.mean(X, axis=2), 2)
            stds = np.expand_dims(np.std(X, axis=2), 2)
            X = (X - means) / stds
            y = (y - means[:, outputs_idx, :]) / stds[:, outputs_idx, :]
            y = np.reshape(y, newshape=(y.shape[0], -1), order='F')

    return X, y

def make_output_relative(y:np.array,
                         X:np.array,
                         reference_idx:int) -> np.array:
    """ Convert each value in y to percentual change relative to most recent
    value in X at index reference_idx.
    E.g. turn stock prices into percentual increase/decrease relative to last
    close.

    Args:
        y: Output values
        X: Input timeseries data
        reference_idx: Index of reference feature

    Returns:
        Converted y
    """
    if len(X.shape) == 0:
        raise ValueError('X must not be an empty array.')
    y_out = deepcopy(y)
    for i in range(X.shape[0]):
        reference_value = X[i][reference_idx, -1]
        y_out[i] = (y_out[i] - reference_value) / reference_value

    return y_out