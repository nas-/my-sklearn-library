import os
import random
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
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


def initialize_devices(device: str) -> None:
    """
    Initialize a TPU or a GPU, depending on imput ==TPU or GPU
    :param device: str, TPU or GPU
    """
    if device == "TPU":
        print("connecting to TPU...")
        try:
            # detect and init the TPU
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU ', tpu.master())
        except ValueError:
            print("Could not connect to TPU")
            tpu = None

        if tpu:
            try:
                print("initializing  TPU ...")
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                # instantiate a distribution strategy
                strategy = tf.distribute.experimental.TPUStrategy(tpu)
                print("TPU initialized")
            except _:
                print("failed to initialize TPU")
        else:
            device = "GPU"

    if device != "TPU":
        print("Using default strategy for CPU and single GPU")
        # instantiate a distribution strategy
        strategy = tf.distribute.get_strategy()

    if device == "GPU":
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    AUTO = tf.data.experimental.AUTOTUNE
    REPLICAS = strategy.num_replicas_in_sync
    print(f'REPLICAS: {REPLICAS}')


class DFdiff(object):
    """
    @url=https://gist.github.com/sanzoghenzo/73275613c592331180a24cb2ddfd5bcb
    Identify differences between two pandas DataFrames using a key column.
    Key column is assumed to have a unique row identifier, i.e. no duplicates.

    :param path1: Path or DataFrame of first dataframe.
    :param path2: Path or DataFrame of first dataframe.
    :param out_path: Path of outfile
    :param index_col_name: index common name

    """

    def __init__(self, path1: Union[str, pd.DataFrame], path2: Union[str, pd.DataFrame], out_path: str,
                 index_col_name: str, **kwargs) -> None:

        self.old_df = self._parse_file(path1, **kwargs)
        self.new_df = self._parse_file(path2, **kwargs)
        self.out_path = out_path
        self.index = index_col_name
        diff = self.diff_pd()
        if diff:
            with pd.ExcelWriter(out_path) as writer:
                for sname, data in diff.items():
                    data.to_excel(writer, sheet_name=sname)
            print(f"Differences saved in {out_path}")
        else:
            print("No differences spotted")

    @staticmethod
    def _parse_file(path, **kwargs):
        if isinstance(path, pd.DataFrame):
            return path
        file_extension = Path(path).suffix
        if file_extension == 'csv':
            func = pd.read_csv
        elif file_extension in ['xlsx']:
            func = pd.read_excel
        else:
            raise ValueError(f'File {file_extension} is not supported')
        return func(path, **kwargs)

    @staticmethod
    def _report_diff(x):
        """Function to use with groupby.apply to highlight value changes."""
        return x[0] if x[0] == x[1] or pd.isna(x).all() else f'{x[0]} ---> {x[1]}'

    @staticmethod
    def _strip(x):
        """Function to use with applymap to strip whitespaces from a dataframe."""
        return x.strip() if isinstance(x, str) else x

    def diff_pd(self):
        # setting the column name as index for fast operations
        old_df = self.old_df.set_index(self.index)
        new_df = self.new_df.set_index(self.index)
        # get the added and removed rows
        old_keys = old_df.index
        new_keys = new_df.index
        if isinstance(old_keys, pd.MultiIndex):
            removed_keys = old_keys.difference(new_keys)
            added_keys = new_keys.difference(old_keys)
        else:
            removed_keys = np.setdiff1d(old_keys, new_keys)
            added_keys = np.setdiff1d(new_keys, old_keys)
        # populate the output data with non empty dataframes
        out_data = {}
        removed = old_df.loc[removed_keys]
        if not removed.empty:
            out_data["removed"] = removed
        added = new_df.loc[added_keys]
        if not added.empty:
            out_data["added"] = added
        # focusing on common data of both dataframes
        common_keys = np.intersect1d(old_keys, new_keys, assume_unique=True)
        common_columns = np.intersect1d(
            old_df.columns, new_df.columns, assume_unique=True
        )
        new_common = new_df.loc[common_keys, common_columns].applymap(self._strip)
        old_common = old_df.loc[common_keys, common_columns].applymap(self._strip)
        # get the changed rows keys by dropping identical rows
        # (indexes are ignored, so we'll reset them)
        common_data = pd.concat(
            [old_common.reset_index(), new_common.reset_index()], sort=True
        )
        changed_keys = common_data.drop_duplicates(keep=False)[self.index]
        if isinstance(changed_keys, pd.Series):
            changed_keys = changed_keys.unique()
        else:
            changed_keys = changed_keys.drop_duplicates().set_index(self.index).index
        # combining the changed rows via multi level columns
        df_all_changes = pd.concat(
            [old_common.loc[changed_keys], new_common.loc[changed_keys]],
            axis='columns',
            keys=['old', 'new']
        ).swaplevel(axis='columns')
        # using report_diff to merge the changes in a single cell with "-->"
        df_changed = df_all_changes.groupby(level=0, axis=1).apply(
            lambda frame: frame.apply(self._report_diff, axis=1))
        # add changed df to output data only if non empty
        if not df_changed.empty:
            out_data['changed'] = df_changed

        return out_data
