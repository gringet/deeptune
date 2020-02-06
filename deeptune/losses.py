import keras.backend as K
import numpy as np


def padded_categorical_crossentropy(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> float:
    """ A padded categorical crossentropy loss function

    Expect the last dimension of y_true and y_pred be one hot encoded vectors.
    Vectors without any one hot entry (all zeroes) are considered as padding.

    # Arguments:
        y_true (np.ndarray): ground truth
        y_pred (np.ndarray): predictions
    # Returns:
        the mean categorical crossentropy
    """
    mask = K.any(y_true, -1)
    mask = K.cast(mask, y_true.dtype)
    loss = K.categorical_crossentropy(y_true, y_pred) * mask
    return K.sum(loss) / K.sum(mask)


def padded_categorical_hinge(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> float:
    """ A padded categorical hinge loss function

    Expect the last dimension of y_true and y_pred be one hot encoded vectors.
    Vectors without any one hot entry (all zeroes) are considered as padding.

    # Arguments:
        y_true (np.ndarray): ground truth
        y_pred (np.ndarray): predictions
    # Returns:
        the mean categorical hinge
    """
    mask = K.any(y_true, axis=-1)
    mask = K.cast(mask, y_true.dtype)
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.sum((1.0 - y_true) * y_pred, axis=-1)
    loss = K.maximum(0.0, neg - pos + 1) * mask
    return K.sum(loss) / K.sum(mask)


def weighted_categorical_crossentropy(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> float:
    weight = K.max(y_true)
    y_true /= weight
    loss = K.categorical_crossentropy(y_true, y_pred) * weight
    return loss
