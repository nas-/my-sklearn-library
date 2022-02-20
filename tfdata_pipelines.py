import tensorflow as tf
import pandas as pd
import re
import numpy as np

# SetAutoTune
AUTOTUNE = tf.data.AUTOTUNE


def build_augmenter(is_labelled):
    """
    @url=https://www.kaggle.com/ipythonx/tf-keras-learning-to-resize-image-for-vit-model
    :param is_labelled: bool
    :return: Callable
    """

    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_saturation(img, 0.65, 1.05)
        img = tf.image.random_brightness(img, 0.05)
        img = tf.image.random_contrast(img, 0.75, 1.05)
        img = tf.image.random_hue(img, 0.05)
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if is_labelled else augment


def build_decoder(is_labelled: bool):
    """
    @url=https://www.kaggle.com/ipythonx/tf-keras-learning-to-resize-image-for-vit-model
    :param is_labelled: bool
    :return: Callable
    """
    SIZE = 512

    def decode(path):
        file_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(file_bytes, channels=3)
        img = tf.image.resize(img, SIZE)
        return tf.divide(img, 255.)

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if is_labelled else decode


def create_dataset(df: pd.DataFrame,
                   batch_size=32,
                   is_labelled=False,
                   augment=False,
                   repeat=False,
                   shuffle=False):
    """
    @url=https://www.kaggle.com/ipythonx/tf-keras-learning-to-resize-image-for-vit-model
    :param df:pd.DataFrame
    :param batch_size:
    :param is_labelled:
    :param augment:
    :param repeat:
    :param shuffle:
    :return:
    """
    decode_fn = build_decoder(is_labelled)
    augmenter_fn = build_augmenter(is_labelled)

    # Create Dataset
    if is_labelled:
        dataset = tf.data.Dataset.from_tensor_slices((df['Id'].values, df['target_value'].values))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(df['Id'].values)

    dataset = dataset.map(decode_fn, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(augmenter_fn, num_parallel_calls=AUTOTUNE) if augment else dataset
    dataset = dataset.repeat() if repeat else dataset
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True) if shuffle else dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def df_to_dataset(dataframe: pd.DataFrame, target=None, shuffle: bool = False, batch_size: int = 32) -> tf.data.Dataset:
    """
  Transforms a tabular dataframe to a dataset
  @url=https://www.kaggle.com/nasil2/keras-tps2201/edit
 :param target: Target/labels columns
  :param dataframe: DataFrame
  :param shuffle: if shuffle or not
  :param batch_size: batch size
  :return: tf.data.Dataset
  """
    df = dataframe.copy()
    df = {key: value[:, tf.newaxis] for key, value in df.items()}
    if target:
        labels = df.pop(target)
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices((dict(df)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds
