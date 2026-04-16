"""Wavelet coefficient thresholding functions."""

import jax.numpy as jnp


def threshold(data, value, mode="soft", substitute=0):
    """Threshold wavelet coefficients.

    Parameters
    ----------
    data : array
        Input data.
    value : float
        Threshold value.
    mode : str
        Thresholding mode: 'soft', 'hard', 'garrote', 'greater', or
        'less'. Default 'soft'.
    substitute : float
        Value to use for thresholded entries. Default 0.

    Returns
    -------
    array
        Thresholded data.
    """
    return {
        "soft": _soft,
        "hard": _hard,
        "garrote": _garrote,
        "greater": _greater,
        "less": _less,
    }[mode](data, value, substitute)


def _soft(data, value, substitute=0):
    mag = jnp.abs(data)
    return jnp.where(
        mag < value, substitute, jnp.sign(data) * jnp.maximum(mag - value, 0)
    )


def _hard(data, value, substitute=0):
    return jnp.where(jnp.abs(data) < value, substitute, data)


def _garrote(data, value, substitute=0):
    mag = jnp.abs(data)
    return jnp.where(
        mag < value, substitute, data * jnp.maximum(1 - (value / mag) ** 2, 0)
    )


def _less(data, value, substitute=0):
    return jnp.where(data > value, substitute, data)


def _greater(data, value, substitute=0):
    return jnp.where(data < value, substitute, data)


def threshold_firm(data, value_low, value_high):
    """Firm (semi-soft) thresholding.

    Parameters
    ----------
    data : array
        Input data.
    value_low : float
        Lower threshold. Coefficients below this are zeroed.
    value_high : float
        Upper threshold. Coefficients above this are kept unchanged.
        Values in between are shrunk linearly.

    Returns
    -------
    array
        Thresholded data.
    """
    mag = jnp.abs(data)
    vdiff = value_high - value_low
    shrunk = data * jnp.clip(value_high * (1 - value_low / mag) / vdiff, 0, None)
    return jnp.where(mag > value_high, data, shrunk)
