from typing import Callable

import tensorflow as tf


def get_normalization_layer(name: str, dataset: tf.data.Dataset) -> tf.keras.Layer:
    """creates a normalization layer for a NUMERICAL feature and adapt it.
    :param dataset: Dataset on which to adapt the layer
    :type name: feature name
    """
    # Create a Normalization layer for the feature.
    normalizer = tf.keras.layers.Normalization(axis=None)

    # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer


def get_category_encoding_layer(name: str, dataset: tf.data.Dataset, dtype: str, max_tokens: int = None) -> Callable:
    """
    Create an encoding layer for a categorical variable
    To use as something like get_category_encoding_layer(...)(categorical_col)
    :param name: Name of the feature
    :param dataset: Dataset
    :param dtype: dtype
    :param max_tokens: Max tokens where to split.
    :return: Callable
    """
    # Create a layer that turns strings into integer indices.
    if dtype == 'string':
        index = tf.keras.layers.StringLookup(max_tokens=max_tokens)
    # Otherwise, create a layer that turns integer values into integer indices.
    else:
        index = tf.keras.layers.IntegerLookup(max_tokens=max_tokens)

    # Prepare a `tf.data.Dataset` that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Encode the integer indices.
    encoder = tf.keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply multi-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))