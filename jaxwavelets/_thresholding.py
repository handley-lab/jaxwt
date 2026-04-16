"""Wavelet coefficient thresholding functions."""

import jax.numpy as jnp


def soft_threshold(data, value, substitute=0):
    """Soft thresholding: shrink coefficients toward zero.

    Parameters
    ----------
    data : array
        Input data.
    value : float
        Threshold value.
    substitute : float
        Value for thresholded entries. Default 0.

    Returns
    -------
    array
        Thresholded data.
    """
    mag = jnp.abs(data)
    return jnp.where(
        mag < value, substitute, jnp.sign(data) * jnp.maximum(mag - value, 0)
    )


def hard_threshold(data, value, substitute=0):
    """Hard thresholding: zero out coefficients below threshold.

    Parameters
    ----------
    data : array
        Input data.
    value : float
        Threshold value.
    substitute : float
        Value for thresholded entries. Default 0.

    Returns
    -------
    array
        Thresholded data.
    """
    return jnp.where(jnp.abs(data) < value, substitute, data)


def garrote_threshold(data, value, substitute=0):
    """Non-negative garrote thresholding: intermediate between soft and hard.

    Parameters
    ----------
    data : array
        Input data.
    value : float
        Threshold value.
    substitute : float
        Value for thresholded entries. Default 0.

    Returns
    -------
    array
        Thresholded data.
    """
    mag = jnp.abs(data)
    return jnp.where(
        mag < value, substitute, data * jnp.maximum(1 - (value / mag) ** 2, 0)
    )


def firm_threshold(data, value_low, value_high):
    """Firm (semi-soft) thresholding: linear interpolation between soft and hard.

    Parameters
    ----------
    data : array
        Input data.
    value_low : float
        Lower threshold. Coefficients below this are zeroed.
    value_high : float
        Upper threshold. Coefficients above this are kept unchanged.

    Returns
    -------
    array
        Thresholded data.
    """
    mag = jnp.abs(data)
    vdiff = value_high - value_low
    shrunk = data * jnp.clip(value_high * (1 - value_low / mag) / vdiff, 0, None)
    return jnp.where(mag > value_high, data, shrunk)
