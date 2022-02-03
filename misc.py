import pandas as pd
import numpy as np
import random
import os
import tensorflow as tf


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True):  # sourcery skip: merge-else-if-into-elif
    """
    Reduce memory usage of a DataFrame

    :type verbose: bool
    :param verbose: If verbose, print reduction
    :type df: pd.DataFrame
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtypes
        # pd.api.types.is_numeric_dtype
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem,
                                                                              100 * (start_mem - end_mem) / start_mem))

    return df


def better_than_median(inputs: np.ndarray, axis: int) -> np.ndarray:
    """
    @url=https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/282735
    Compute the mean of the predictions if there are no outliers,
    or the median if there are outliers.
    :param inputs: ndarray of shape (n_samples, n_folds)
    :param axis: Axis 1 or axis 0
    :return: """
    spread = inputs.max(axis=axis) - inputs.min(axis=axis)
    spread_lim = 0.45
    print(f"Inliers:  {(spread < spread_lim).sum():7} -> compute mean")
    print(f"Outliers: {(spread >= spread_lim).sum():7} -> compute median")
    print(f"Total:    {len(inputs):7}")
    return np.where(spread < spread_lim,
                    np.mean(inputs, axis=axis),
                    np.median(inputs, axis=axis))


def seeding(SEED: int, use_tf: bool = False) -> None:
    """
    Utility function to set all random seeds
    :param SEED: seed to set
    :param use_tf: whether to set seed fot tensorflow or not
    """
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(SEED)
    if use_tf:
        tf.random.set_seed(SEED)
    print('seeding done!!!')
