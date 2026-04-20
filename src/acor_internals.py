from __future__ import annotations

import numpy as np

VALID_METHODS = {"akc", "agc", "cid", "cma"}
VALID_ALTERNATIVES = {"two.sided", "less", "greater"}
VALID_VARIANCE_METHODS = {"delta", "ij"}


def validate_inputs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Validate and normalise *x* and *y*.

    Returns
    -------
    x : ndarray, shape (n, m)
    y : ndarray, shape (n,)
    n : int
    m : int
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a 1-D array.")

    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    elif x.ndim != 2:
        raise ValueError("x must be a 1-D or 2-D array.")

    n = y.shape[0]
    if x.shape[0] != n:
        raise ValueError("x and y must have the same number of observations.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("NaN values not supported; remove them first.")
    if len(np.unique(y)) < 2:
        raise ValueError("y must have at least 2 distinct values.")

    m = x.shape[1]
    return x, y, n, m


def validate_method(method: str) -> str:
    method = method.lower()
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS!r}, got {method!r}")
    return method


def validate_alternative(alternative: str) -> str:
    alternative = alternative.lower()
    if alternative not in VALID_ALTERNATIVES:
        raise ValueError(
            f"alternative must be one of {VALID_ALTERNATIVES!r}, got {alternative!r}"
        )
    return alternative


def validate_conf_level(conf_level: float) -> float:
    if isinstance(conf_level, bool) or not isinstance(conf_level, (int, float, np.floating)):
        raise ValueError("conf_level must be a numeric value.")
    conf_level = float(conf_level)
    if not 0.0 < conf_level < 1.0:
        raise ValueError("conf_level must be strictly between 0 and 1.")
    return conf_level


def validate_variance_method(variance: str) -> str:
    variance = variance.lower()
    if variance not in VALID_VARIANCE_METHODS:
        raise ValueError(
            f"variance must be one of {VALID_VARIANCE_METHODS!r}, got {variance!r}"
        )
    return variance
